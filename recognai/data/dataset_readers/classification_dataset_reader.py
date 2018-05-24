import os
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, Field
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

from recognai.data.dataset_readers.classification_instance_preparator import ClassificationInstancePreparator
from recognai.data.readers.es_reader import *
from recognai.data.readers.file_reader import *
from recognai.data.readers.helpers import *
from recognai.data.tokenizer.word_splitter import SpacyWordSplitter

__name__ = "classification_dataset_reader"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TOKENS_FIELD = 'tokens'
LABEL_FIELD = 'gold_label'
DEFAULT_DASK_CLUSTER = '127.0.0.1:8786'


@DatasetReader.register(__name__)
class ClassificationDatasetReader(DatasetReader):
    def __init__(self,
                 dataset_transformations: Dict,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 dask_cluster: str = DEFAULT_DASK_CLUSTER,

                 block_size: int = 10e6,
                 cache_size: int = None) -> None:

        super(ClassificationDatasetReader, self).__init__(lazy=True)

        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
        self._token_indexers = token_indexers
        self._instance_preparator = ClassificationInstancePreparator(dataset_transformations)
        self._reader_block_size = block_size

        from dask.distributed import Client
        from dask.cache import Cache
        if cache_size:
            cache = Cache(cache_size)
            cache.register()

        self._client=None
        try:
            self._client = dask.distributed.Client(dask_cluster)
        except:
            self._client = dask.distributed.Client()

    def __del__(self):
        try:
            self._client.close()
        except:
            ''''''

    def process_example(self, example: Dict) -> Instance:
        example = self._instance_preparator.read_info(example)
        logger.debug('Example:[%s]', example)
        return self.text_to_instance(example)

    def text_to_instance(self,  # type: ignore
                         example: Dict) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        gold_label_id = self._instance_preparator.get_gold_label_id()

        for field, value in example.items():
            if field == gold_label_id:
                fields[gold_label_id] =  LabelField(value)
            else:
                input_tokens = self._tokenizer.tokenize(value)
                fields[field] = TextField(input_tokens, self._token_indexers)
        return Instance(fields)

    @overrides
    def _read(self, dataset_config: Any) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache

        if isinstance(dataset_config, str):
            dataset_config = {'path': dataset_config}
        elif isinstance(dataset_config, Params):
            dataset_config = dataset_config.as_dict()

        logger.info("Reading instances from dataset at: %s", dataset_config)
        dataset = self.__build_dataset(dataset_config).persist()
        logger.info("Finished reading instances")

        for example in dataset:
            yield self.process_example(example)

    def __build_dataset(self, config: Dict) -> Bag:
        params = config.copy()  # Preserve original config (multiple reads)

        if is_elasticsearch_source(params):
            return from_elasticsearch(**params)

        if not 'format' in params:
            _, extension = os.path.splitext(params['path'])
            params['format'] = extension[1:]

        format: str = params.pop('format', JSON_FORMAT)
        if is_json(format):
            return from_json(**params, blocksize=self._reader_block_size)
        else:
            return from_csv(**params, blocksize=self._reader_block_size, assume_missing=True)

    @classmethod
    def from_params(cls, params: Params) -> 'ClassificationDatasetReader':

        block_size = params.pop('block_size', 10e6)  # Deafault 10MB
        cache_size = params.pop('cache_size', 1e9)  # Default 1GB

        _ = params.pop('dataset_format', {})  # Backward compatibility
        _ = params.pop('connection', {})  # Backward compatibility
        ds_transformations = params.pop('transformations', dict()).as_dict()

        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        dask_cluster = params.pop('dask_cluster', DEFAULT_DASK_CLUSTER)

        params.assert_empty(cls.__name__)
        return ClassificationDatasetReader(dataset_transformations=ds_transformations,
                                           token_indexers=token_indexers,
                                           dask_cluster=dask_cluster,
                                           block_size=block_size,
                                           cache_size=cache_size)


@DatasetReader.register("parallel_dataset_reader")
class ParallelDatasetReader(ClassificationDatasetReader):
    """Just for backward compatibility"""
