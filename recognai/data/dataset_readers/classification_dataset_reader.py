from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, Field
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

from recognai.data.sources.file import *
from recognai.data.sources.helpers import *
from recognai.data.sources.helpers import read_dataset
from recognai.data.tokenizer.word_splitter import SpacyWordSplitter

__name__ = "classification_dataset_reader"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_GOLD_LABEL_ID = 'gold_label'

@DatasetReader.register(__name__)
class ClassificationDatasetReader(DatasetReader):
    def __init__(self,
                 target_field: str,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        super(ClassificationDatasetReader, self).__init__(lazy=True)

        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
        self._token_indexers = token_indexers
        self._target_field = target_field

    def process_example(self, example: Dict) -> Instance:
        logger.debug('Example:[%s]', example)
        return self.text_to_instance(example)

    def text_to_instance(self,  # type: ignore
                         example: Dict) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        gold_label_id = self._target_field

        for field, value in example.items():
            tensor = LabelField(value) \
                if field == gold_label_id \
                else TextField(self._tokenizer.tokenize(value), self._token_indexers)
            fields[field] = tensor

        return Instance(fields)

    @overrides
    def _read(self, dataset_config: Any) -> Iterable[Instance]:
        for example in read_dataset(dataset_config):
            yield self.process_example(example)

    @classmethod
    def from_params(cls, params: Params) -> 'ClassificationDatasetReader':

        _ = params.pop('dataset_format', {})  # Backward compatibility
        _ = params.pop('connection', {})  # Backward compatibility
        _ = params.pop('transformations', dict()).as_dict()  # Backward compatibility
        _ = Tokenizer.from_params(params.pop('tokenizer', {}))

        target_field = params.pop('target', DEFAULT_GOLD_LABEL_ID)
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))

        params.assert_empty(cls.__name__)
        return ClassificationDatasetReader(target_field=target_field, token_indexers=token_indexers)


@DatasetReader.register("parallel_dataset_reader")
class ParallelDatasetReader(ClassificationDatasetReader):
    """Just for backward compatibility"""
