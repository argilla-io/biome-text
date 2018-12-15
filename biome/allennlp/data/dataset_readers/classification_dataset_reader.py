from allennlp.data import DatasetReader, Instance, TokenIndexer, Field
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

from biome.allennlp.data.tokenizer.word_splitter import SpacyWordSplitter
from biome.data.sources.file import *
from biome.data.sources.helpers import *
from biome.data.utils import read_datasource_cfg

__name__ = "classification_dataset_reader"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_GOLD_LABEL_ID = 'gold_label'


def _text_to_instance(example, gold_label_id, token_indexers, tokenizer):
    fields: Dict[str, Field] = {}
    for field, value in example.items():
        if not is_reserved_field(field):
            tensor = LabelField(value) \
                if field == gold_label_id \
                else TextField(tokenizer.tokenize(value), token_indexers)
            fields[field] = tensor
    return Instance(fields)


@DatasetReader.register(__name__)
class ClassificationDatasetReader(DatasetReader):
    def __init__(self,
                 target: str = DEFAULT_GOLD_LABEL_ID,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        super(ClassificationDatasetReader, self).__init__(lazy=True)

        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
        self._token_indexers = token_indexers
        self._target_field = target
        self._cached_datasets = dict()

    def process_example(self, example: Dict) -> Instance:
        if example:
            logger.debug('Example:[%s]', example)
            return self.text_to_instance(example)

    def text_to_instance(self,  # type: ignore
                         example: Dict) -> Instance:
        # pylint: disable=arguments-differ

        gold_label_id = self._target_field
        tokenizer = self._tokenizer
        token_indexers = self._token_indexers

        return _text_to_instance(example, gold_label_id, token_indexers, tokenizer)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        cfg = read_datasource_cfg(file_path)

        ds_key = id(cfg)
        if not self._cached_datasets.get(ds_key):
            logger.debug('Read dataset from {}'.format(file_path))
            self._cached_datasets[ds_key] = read_dataset(cfg)

        dataset = self._cached_datasets[ds_key]
        logger.debug('Loaded from cache dataset {}'.format(file_path))

        return (self.process_example(example) for example in dataset)
