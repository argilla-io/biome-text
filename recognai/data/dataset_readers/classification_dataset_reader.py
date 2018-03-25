import json
import logging
from typing import Iterable, Dict, Any

import dask.bag as db
import dask.dataframe as dd
from allennlp.common import Params
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, Field
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from dask.bag import Bag
from overrides import overrides

from recognai.data.dataset_readers.reader_utils import is_json
from recognai.data.dataset_readers.classification_instance_preparator import ClassificationInstancePreparator
from recognai.data.dataset_readers.reader_utils import CsvConfig
from recognai.data.tokenizer.word_splitter import SpacyWordSplitter

__name__ = "classification_dataset_reader"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TOKENS_FIELD = 'tokens'
LABEL_FIELD = 'label'


@DatasetReader.register(__name__)
class ClassificationDatasetReader(DatasetReader):
    def __init__(self,
                 ds_format: Any,
                 dataset_transformations: Dict,
                 storage_options: Dict[str, str],
                 token_indexers: Dict[str, TokenIndexer] = None,

                 block_size: int = 10e6,
                 cache_size: int = None,
                 ) -> None:

        super(ClassificationDatasetReader, self).__init__(lazy=True)

        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
        self._token_indexers = token_indexers or {TOKENS_FIELD: SingleIdTokenIndexer()}
        self._instance_preparator = ClassificationInstancePreparator(dataset_transformations)

        from dask.cache import Cache

        if cache_size:
            cache = Cache(cache_size)
            cache.register()

        self._storage_options = storage_options
        self._reader_block_size = block_size
        self._ds_format = 'json' if is_json(ds_format) else CsvConfig.from_params(ds_format)

    def read_csv_dataset(self, path: str) -> Iterable[Bag]:
        dataframe = dd.read_csv(path,
                                blocksize=self._reader_block_size,
                                assume_missing=True,
                                sep=self._ds_format._delimiter,
                                storage_options=self._storage_options)
        columns = [column.strip() for column in  dataframe.columns]  # csv meta data
        return [dataframe
                    .persist()
                    .to_bag(index=False)
                    .map(lambda row: dict(zip(columns, row)))
                    .persist()]

    def read_json_dataset(self, path: str) -> Iterable[Bag]:
        return [db.read_text(path, blocksize=self._reader_block_size)
                    .persist()
                    .map(json.loads)
                    .persist()]

    def process_example(self, example: Dict) -> Instance:
        input_text, label = self._instance_preparator.read_info(example)
        logger.debug('Example:[%s];Input:[%s];Label:[%s]', example, input_text, label)
        return self.text_to_instance(input_text, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         input_text: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        input_tokens = self._tokenizer.tokenize(input_text)
        fields[TOKENS_FIELD] = TextField(input_tokens, self._token_indexers)

        if label:
            fields[LABEL_FIELD] = LabelField(label)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache

        logger.info("Reading instances from dataset at: %s", file_path)
        partitions = self.read_json_dataset(file_path) if is_json(self._ds_format) else self.read_csv_dataset(file_path)
        logger.info("Finished reading instances")

        for partition in partitions:
            for example in partition:
                yield self.process_example(example)

    @classmethod
    def from_params(cls, params: Params) -> 'ClassificationDatasetReader':

        block_size = params.pop('block_size', 10e6)  # Deafault 10MB
        cache_size = params.pop('cache_size', 1e9)  # Default 1GB

        ds_transformations = params.pop('transformations', dict()).as_dict()
        storage_options = params.pop('connection', dict()).as_dict()

        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))

        ds_format = params.pop('dataset_format', None)

        params.assert_empty(cls.__name__)
        return ClassificationDatasetReader(ds_format,
                                           ds_transformations,
                                           storage_options,
                                           token_indexers,
                                           block_size=block_size, cache_size=cache_size)


@DatasetReader.register("parallel_dataset_reader")
class ParallelDatasetReader(ClassificationDatasetReader):
    """Just for backward compatibility"""
