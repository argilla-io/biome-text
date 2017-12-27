from typing import Dict, Iterable

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

__name__ = "generic_classification_reader"


class CsvConfig(object):
    def __init__(self, delimiter: str):
        self._delimiter = delimiter

    @classmethod
    def from_params(cls, params: Params) -> 'CsvConfig':
        delimiter = params.pop('delimiter')
        return CsvConfig(delimiter)


def _read_csv_file(config: CsvConfig):
    def inner(input_file: str) -> Iterable[Dict]:
        with open(input_file) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=config._delimiter)

            for example in reader:
                logger.info(example)
                yield example

    return inner


def _read_json_file(input_file: str):
    for line in tqdm.tqdm(input_file):
        yield json.loads(line)


FILE_FORMATTERS = {
    'json': _read_json_file,
    'csv': _read_csv_file
}

INPUTS_FIELD = 'inputs'
GOLD_LABEL = 'gold_label'

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _read_from_file(format_file: str) -> Dict:
    with open(format_file) as format_config:
        return json.loads(format_config.read())


class GenericClassificationReader(DatasetReader):
    __TOKENS_FIELD = 'tokens'
    __LABEL_FIELD = 'label'

    def __init__(self,
                 file_formatter,  # TODO mark as function
                 dataset_format: Dict,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        self._tokenizer = tokenizer or WordTokenizer()
        self._dataset_format = dataset_format
        self._file_formatter = file_formatter
        self._token_indexers = token_indexers or {GenericClassificationReader.__TOKENS_FIELD: SingleIdTokenIndexer()}

    """
    Reads a file from a classification dataset.  This data is
    formatted as csv. We convert these keys into fields named "label", "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def _input(self, example: Dict) -> str:
        inputs_field = self._dataset_format[INPUTS_FIELD]
        return inputs_field \
            if type(inputs_field) is str \
            else  " ".join([example[input] for input in inputs_field])

    def _gold_label(self, example: Dict) -> str:

        def with_mapping(value, mapping=None):
            return mapping.get(value) if mapping else value

        field_type = "field"
        field_mapping = "values_mapping"

        gold_label_definition = self._dataset_format[GOLD_LABEL]

        return example[gold_label_definition] \
            if type(gold_label_definition) is str \
            else with_mapping(example[gold_label_definition[field_type]], gold_label_definition.get(field_mapping))

    @overrides
    def read(self, file_path: str) -> Dataset:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        with open(file_path, 'r') as input_file:
            logger.info("Reading instances from dataset at: %s", file_path)

            for example in self._file_formatter(file_path):
                input_text = self._input(example)
                label = self._gold_label(example)

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
        fields[GenericClassificationReader.__TOKENS_FIELD] = TextField(input_tokens, self._token_indexers)

        if label:
            fields[GenericClassificationReader.__LABEL_FIELD] = LabelField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'GenericClassificationReader':

        dataset_format = params.pop('format', dict())
        file_formatter = FILE_FORMATTERS['csv'](CsvConfig(delimiter=','))  # TODO calculate from input

        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))

        params.assert_empty(cls.__name__)

        return GenericClassificationReader(
            file_formatter=file_formatter,
            dataset_format=dataset_format,
            tokenizer=tokenizer,
            token_indexers=token_indexers)
