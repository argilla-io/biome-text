from typing import Any

from allennlp.data import DatasetReader, Instance, TokenIndexer, Field, Tokenizer
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


def _text_to_instance(example: Dict,
                      forward_definition: Dict[str, Any],
                      gold_label_id: str,
                      token_indexers: Dict[str, TokenIndexer],
                      tokenizer: Tokenizer):
    def instance_by_forward_definition(example: Dict,
                                       forward_definition: Dict[str, Any],
                                       token_indexers: Dict[str, TokenIndexer],
                                       tokenizer: Tokenizer
                                       ) -> Instance:
        def field_from_type(type: str, value: Any) -> Field:
            cls = globals()[type]
            if cls == LabelField:
                return cls(value)
            if cls == TextField:
                return TextField(tokenizer.tokenize(value), token_indexers)

        fields = {
            field: field_from_type(type, example[field])
            for field, type in forward_definition.items()
            if example.get(field) is not None
        }

        return Instance(fields)

    def instance_by_target_definition(example: Dict,
                                      gold_label_id: str,
                                      token_indexers: Dict[str, TokenIndexer],
                                      tokenizer: Tokenizer
                                      ) -> Instance:
        fields: Dict[str, Field] = {}

        for field, value in example.items():
            if not is_reserved_field(field):
                tensor = LabelField(value) \
                    if field == gold_label_id \
                    else TextField(tokenizer.tokenize(value), token_indexers)
                fields[field] = tensor

        return Instance(fields)

    return instance_by_forward_definition(example, forward_definition, token_indexers, tokenizer) \
        if forward_definition \
        else instance_by_target_definition(example, gold_label_id, token_indexers, tokenizer)


@DatasetReader.register(__name__)
class ClassificationDatasetReader(DatasetReader):

    def __init__(self,
                 forward: Dict[str, Any] = None,
                 target: str = DEFAULT_GOLD_LABEL_ID,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        super(ClassificationDatasetReader, self).__init__(lazy=True)

        self.__tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
        self.__token_indexers = token_indexers
        self.__target_field = target
        self.__forward_definition = forward
        self.__cached_datasets = dict()

    def process_example(self, example: Dict) -> Instance:
        if example:
            logger.debug('Example:[%s]', example)
            return self.text_to_instance(example)

    def text_to_instance(self,  # type: ignore
                         example: Dict) -> Instance:
        # pylint: disable=arguments-differ

        gold_label_id = self.__target_field
        tokenizer = self.__tokenizer
        token_indexers = self.__token_indexers
        forward_definition = self.__forward_definition

        return _text_to_instance(example, forward_definition, gold_label_id, token_indexers, tokenizer)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        cfg = read_datasource_cfg(file_path)

        ds_key = id(cfg)
        if not self.__cached_datasets.get(ds_key):
            logger.debug('Read dataset from {}'.format(file_path))
            self.__cached_datasets[ds_key] = read_dataset(cfg)

        dataset = self.__cached_datasets[ds_key]
        logger.debug('Loaded from cache dataset {}'.format(file_path))

        return (self.process_example(example) for example in dataset)
