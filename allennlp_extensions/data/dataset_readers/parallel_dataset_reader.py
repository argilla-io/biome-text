import json
import logging

from allennlp.common import Params
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from dask.bag import Bag
from dask.dataframe import DataFrame
from overrides import overrides
from typing import Iterable, Callable, Dict, Any
from allennlp.common.file_utils import cached_path

import dask.dataframe as dd
import dask.bag as db

from allennlp_extensions.data.dataset_readers import ClassificationDatasetReader
from allennlp_extensions.data.dataset_readers.classification_dataset_reader import is_json
import dask.multiprocessing

from multiprocessing.pool import ThreadPool

from allennlp_extensions.data.dataset_readers.reader_utils import CsvConfig

__name__ = "parallel_dataset_reader"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register(__name__)
class ParallelDatasetReader(ClassificationDatasetReader):
    def __init__(self,
                 ds_format: Any,
                 dataset_transformations: Dict,
                 storage_options: Dict[str, str],
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,

                 block_size: int = 10e6,
                 cache_size: int = None,
                 ) -> None:
        super(ParallelDatasetReader, self).__init__(
            file_reader=None,
            token_indexers=token_indexers,
            tokenizer=tokenizer,
            dataset_transformations=dataset_transformations
        )

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
        columns = dataframe.columns  # csv meta data
        return [dataframe
                    .persist()
                    .to_bag(index=False)
                    .map(lambda row: dict(zip(columns, row)))
                    .persist()]
        # for partition_idx in range(0, dataframe.npartitions)]

    def read_json_dataset(self, path: str) -> Iterable[Bag]:
        return [db.read_text(path, blocksize=self._reader_block_size)
                    .persist()
                    .map(json.loads)
                    .persist()]

    # for partition_idx in range(0, dataframe.npartitions)]

    def process_example(self, example: Dict) -> Instance:
        input_text, label = self.read_info(example)
        logger.debug('Example:[%s];Input:[%s];Label:[%s]', example, input_text, label)
        return self.text_to_instance(input_text, label)

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
    def from_params(cls, params: Params) -> 'ParallelDatasetReader':

        block_size = params.pop('block_size', 10e6)  # Deafault 10MB
        cache_size = params.pop('cache_size', 1e9)  # Default 1GB

        dataset_format = params.pop('transformations', dict()).as_dict()
        storage_options = params.pop('connection', dict()).as_dict()

        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))

        format = params.pop('dataset_format', None)

        params.assert_empty(cls.__name__)
        return ParallelDatasetReader(format,
                                     dataset_format,
                                     storage_options,
                                     tokenizer, token_indexers,
                                     block_size=block_size, cache_size=cache_size)
