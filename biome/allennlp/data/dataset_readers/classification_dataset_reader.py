from typing import Any, Dict, Iterable

from allennlp.data import DatasetReader, Instance, TokenIndexer, Field, Tokenizer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

from biome.allennlp.commands.helpers import read_datasource_configuration
from biome.allennlp.data.tokenizer.word_splitter import SpacyWordSplitter
from biome.data.sources import RESERVED_FIELD_PREFIX
from biome.data.sources.helpers import logging, read_dataset

__name__ = "classification_dataset_reader"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_GOLD_LABEL_ID = 'gold_label'


@DatasetReader.register(__name__)
class ClassificationDatasetReader(DatasetReader):
    """A DatasetReader for a Classification dataset

    Parameters
    ----------
    forward
    target
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter
    token_indexers
    """

    def __init__(self,
                 forward: Dict[str, Any] = None,
                 target: str = DEFAULT_GOLD_LABEL_ID,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        super(ClassificationDatasetReader, self).__init__(lazy=True)

        self.__tokenizer = tokenizer or WordTokenizer(word_splitter=SpacyWordSplitter())
        self.__token_indexers = token_indexers
        self.__target_field = target
        self.__forward_definition = forward
        self.__cached_datasets = dict()

    def process_example(self, example: Dict) -> Instance:
        logger.debug('Example:[%s]', example)
        try:
            return self.text_to_instance(example)
        except Exception as e:
            logger.warning(e)
            return Instance({})  # An empty Instance({}) resolves to False in an if statements

    def text_to_instance(self,  # type: ignore
                         example: Dict) -> Instance:
        # pylint: disable=arguments-differ

        gold_label_id = self.__target_field
        tokenizer = self.__tokenizer
        token_indexers = self.__token_indexers
        forward_definition = self.__forward_definition

        return self.__text_to_instance(example, forward_definition, gold_label_id, token_indexers, tokenizer)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        config = read_datasource_configuration(file_path)
        ds_key = id(config)
        if not self.__cached_datasets.get(ds_key):
            logger.debug('Read dataset from {}'.format(file_path))
            self.__cached_datasets[ds_key] = read_dataset(config).persist()

        dataset = self.__cached_datasets[ds_key]
        logger.debug('Loaded from cache dataset {}'.format(file_path))

        def instance_generator():
            for example in dataset:
                instance = self.process_example(example)
                if instance:
                    yield instance

        return instance_generator()

    @staticmethod
    def __text_to_instance(example: Dict,
                           forward_definition: Dict[str, Any],
                           gold_label_id: str,
                           token_indexers: Dict[str, TokenIndexer],
                           tokenizer: Tokenizer) -> Instance:
        def is_reserved_field(field_name: str) -> bool:
            return field_name and field_name.startswith(RESERVED_FIELD_PREFIX)

        def instance_by_forward_definition() -> Instance:

            def field_from_type(field_type: str, field_value: Any) -> Field:
                if field_type == 'LabelField':
                    return LabelField(field_value)
                elif field_type == 'TextField':
                    return TextField(tokenizer.tokenize(field_value), token_indexers)
                else:
                    raise TypeError(f"{field_type} is not supported yet.")

            fields = {
                field: field_from_type(field_type, example[field])
                for field, field_type in forward_definition.items()
                if example.get(field) is not None
            }

            return Instance(fields)

        def instance_by_target_definition() -> Instance:
            logger.warning("Call to the deprecated method instance_by_target_definition(). "
                           "Use forward definition in config file instead of target.")
            fields: Dict[str, Field] = {}

            for field, value in example.items():
                if not is_reserved_field(field):
                    tensor = LabelField(value) \
                        if field == gold_label_id \
                        else TextField(tokenizer.tokenize(value), token_indexers)
                    fields[field] = tensor

            return Instance(fields)

        return instance_by_forward_definition() \
            if forward_definition \
            else instance_by_target_definition()
