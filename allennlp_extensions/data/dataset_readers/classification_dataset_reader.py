from typing import Dict, Iterable, Any, Callable

import logging

import tqdm
import csv
import json

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from allennlp_extensions.data.tokenizer.word_splitter import SpacyWordSplitter

__name__ = "classification_dataset_reader"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _CsvConfig(object):
    __DEFAULT_DELIMITER = ','

    def __init__(self, delimiter: str):
        self._delimiter = delimiter

    @classmethod
    def from_params(cls, params: Params) -> '_CsvConfig':
        delimiter = params.pop('delimiter', _CsvConfig.__DEFAULT_DELIMITER)
        return _CsvConfig(delimiter)


def _read_from_file(format_file: str) -> Dict:
    with open(format_file) as format_config:
        return json.loads(format_config.read())


def _configure_file_reader(params: Params) -> Callable[[str], Iterable[Dict]]:
    def is_json(format: Any) -> bool:

        format_type = 'json' if not format \
            else str(format).lower() if type(format) is str \
            else str(format['type']).lower()

        return 'json' == format_type

    def csv_file_reader(config: _CsvConfig):
        def inner(input_file: str) -> Iterable[Dict]:
            with open(input_file) as csv_file:
                header = [h.strip()
                          for h in csv_file.readline().split(config._delimiter)]
                reader = csv.DictReader(
                    csv_file, delimiter=config._delimiter, fieldnames=header, skipinitialspace=True)
                for example in reader:
                    yield example

        return inner

    def json_file_reader(input_file: str):
        with open(input_file) as json_file:
            for line in json_file:
                example = json.loads(line)
                logger.debug('Read %s', example)
                yield example

    format = params.pop('dataset_format', None)
    return json_file_reader if is_json(format) else csv_file_reader(_CsvConfig.from_params(format))


@DatasetReader.register(__name__)
class ClassificationDatasetReader(DatasetReader):
    """
    Configuration examples
    ---------------------

    From jsonl files

        {
          "dataset_reader": {
            "type": "classification_dataset_reader",
            "dataset_format": "json",
            "transformations": {
              "inputs": [
                "reviewText"
              ],
              "gold_label": {
                "field": "overall",
                "values_mapping": {
                  "1": "NEGATIVE",
                  "2": "NEGATIVE",
                  "3": "NEUTRAL",
                  "4": "POSITIVE",
                  "5": "POSITIVE"
                }
              }
            },
            "token_indexers": {
              "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
              }
            }
          }
        }

    From csv files:
        {
          "dataset_reader": {
            "type": "classification_dataset_reader",
            "dataset_format": {
              "type": "csv",
              "delimiter" : ","
            },
            "transformations": {
              "inputs": [
                "text"
              ],
              "gold_label": {
                "field": "topic"
              }
            },
            "tokenizer": {
              "word_splitter": {
                "language": "en_core_web_sm"
              }
            },
            "token_indexers": {
              "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
              }
            }
          }
        }
    """

    _TOKENS_FIELD = 'tokens'
    _LABEL_FIELD = 'label'

    __INPUTS_FIELD = 'inputs'
    __GOLD_LABEL = 'gold_label'

    __MISSING_LABEL_DEFAULT = 'None'

    def __init__(self,
                 file_reader: Callable[[str], Iterable[Dict]],
                 dataset_transformations: Dict,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
        self._dataset_transformations = dataset_transformations
        self._file_reader = file_reader
        self._token_indexers = token_indexers or {
            ClassificationDatasetReader._TOKENS_FIELD: SingleIdTokenIndexer()}

    """
    Reads a file from a classification dataset.  This data is
    formatted as csv. We convert these keys into fields named "label", "premise" and "hypothesis".

    Parameters
    ----------
    file_reader: ``Callable[[str], Iterable[Dict]]``
        Define a function to parse input dataset as a collection of dictionaries
    dataset_transformations: ``Dict``
        Define field for input and gold_label from dataset schema 
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    See :class:`TokenIndexer`.
            
    """

    def _input(self, example: Dict) -> str:
        inputs_field = self._dataset_transformations[ClassificationDatasetReader.__INPUTS_FIELD]
        return inputs_field \
            if type(inputs_field) is str \
            else " ".join([example[input] for input in inputs_field])

    def _gold_label(self, example: Dict) -> str:

        def with_mapping(value, mapping=None):
            # Adding default value to value, enables partial mapping
            # Handling missing labels with a default value
            if(value == ""):
                value = ClassificationDatasetReader.__MISSING_LABEL_DEFAULT
            return mapping.get(value, value) if mapping else value

        field_type = "field"
        field_mapping = "values_mapping"

        gold_label_definition = self._dataset_transformations[
            ClassificationDatasetReader.__GOLD_LABEL]

        return str(example[gold_label_definition]
                   if type(gold_label_definition) is str
                   else with_mapping(example[gold_label_definition[field_type]],
                                     gold_label_definition.get(field_mapping, None)))

    @overrides
    def read(self, file_path: str) -> Dataset:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        instances = []

        with open(file_path, 'r') as input_file:
            logger.info("Reading instances from dataset at: %s", file_path)

            for example in self._file_reader(file_path):
                input_text = self._input(example)
                label = self._gold_label(example)

                logger.debug(
                    'Example:[%s];Input:[%s];Label:[%s]', example, input_text, label)
                instances.append(self.text_to_instance(input_text, label))

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         input_text: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        input_tokens = self._tokenizer.tokenize(input_text)
        fields[ClassificationDatasetReader._TOKENS_FIELD] = TextField(
            input_tokens, self._token_indexers)

        if label:
            fields[ClassificationDatasetReader._LABEL_FIELD] = LabelField(
                label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ClassificationDatasetReader':

        file_reader = _configure_file_reader(params)
        dataset_format = params.pop('transformations', dict()).as_dict()

        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))

        params.assert_empty(cls.__name__)

        return ClassificationDatasetReader(
            file_reader=file_reader,
            dataset_transformations=dataset_format,
            tokenizer=tokenizer,
            token_indexers=token_indexers)
