import inspect
import logging
from inspect import Parameter
from typing import Iterable, Dict, Union, Optional, List

from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import TextField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer, SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from biome.data.sources import DataSource
from biome.text.pipelines._impl.allennlp.dataset_readers import CacheableMixin
from biome.text.pipelines._impl.allennlp.dataset_readers import (
    RmSpacesTransforms,
    TextTransforms,
)
from dask.dataframe import Series as DaskSeries


class DataSourceReader(DatasetReader, CacheableMixin):
    """
    A DataSetReader as base for read instances from ``DataSource``

    The subclasses must implements their own way to transform input data to ``Instance``
    in the text_to_field method

    Parameters
    ----------
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter
    token_indexers
        By default we use the following dict {'tokens': SingleIdTokenIndexer}
    segment_sentences
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
    as_text_field
        Flag indicating how to generate the ``TextField``. If enabled, the output Field
        will be a ``TextField`` with text concatenation, else the result field will be
        a ``ListField`` of ``TextField``s, one per input data value
    skip_empty_tokens
        Should i silently skip empty tokens?
    max_sequence_length
        If you want to truncate the text input to a maximum number of characters
    max_nr_of_sentences
        Use only the first max_nr_of_sentences when segmenting the text into sentences
    text_transforms
        By default we use the as 'rm_spaces' registered class, which just removes useless, leading and trailing spaces
        from the text before embedding it in a `TextField`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        segment_sentences: Union[bool, SentenceSplitter] = False,
        as_text_field: bool = True,
        skip_empty_tokens: bool = False,
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
        text_transforms: TextTransforms = None,
    ) -> None:
        DatasetReader.__init__(self, lazy=True)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._sentence_segmenter = segment_sentences
        if segment_sentences is True:
            self._sentence_segmenter = SpacySentenceSplitter()
        self._as_text_field = as_text_field
        self._skip_empty_tokens = skip_empty_tokens
        self._max_sequence_length = max_sequence_length
        self._max_nr_of_sentences = max_nr_of_sentences
        self._text_transforms = text_transforms or RmSpacesTransforms()

        self._logger = logging.getLogger(self.__class__.__name__)

        if segment_sentences and not as_text_field:
            self._logger.warning(
                "You cannot segment sentences and set as_text_field to false at the same time, "
                "i will set as_text_field for the single sentences to true."
            )

        self._signature = {
            name: dict(optional=value.default != Parameter.empty)
            for name, value in inspect.signature(
                self.text_to_instance
            ).parameters.items()
            if name != "self"
        }

    @property
    def signature(self) -> dict:
        """
        Describe de input signature for the pipeline predictions

        Returns
        -------
            A list of expected inputs with information about if input is optional or nor.

            For example, for the signature
            >>def text_to_instance(a:str,b:str, c:str=None)

            This method will return:
            >>{"a":{"optional": False},"b":{"optional": False},"c":{"optional": True}}
        """
        return self._signature.copy()

    def _read(self, file_path: str) -> Iterable[Instance]:
        """An generator that yields `Instance`s that are fed to the model

        This method is implicitly called when training the model.
        The predictor uses the `self.text_to_instance_with_data_filter` method.

        Parameters
        ----------
        file_path
            Path to the configuration file (yml) of the data source.

        Yields
        ------
        instance
            An `Instance` that is fed to the model
        """
        data_source = DataSource.from_yaml(file_path)

        # get cached instances of the data set
        instances = self.get(file_path)
        if instances is not None:
            self._logger.debug("Loaded cached data set %s", file_path)
        else:
            self._logger.debug("Read data set from %s", file_path)
            dataset = data_source.to_mapped_dataframe()
            instances: DaskSeries = dataset.apply(
                lambda x: self.text_to_instance(**x.to_dict()),
                axis=1,
                meta=(None, "object"),
            )

            # cache instances of the data set
            self.set(file_path, instances)

        return (instance for _, instance in instances.iteritems() if instance)

    def _value_as_string(self, value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return self._value_as_string(value.values())
        if isinstance(value, Iterable):
            return " ".join(map(self._value_as_string, value))
        return str(value)

    def build_textfield(
        self, data: Union[str, Iterable, dict]
    ) -> Optional[Union[ListField, TextField]]:
        """Embeds the record in a TextField or ListField depending on the _as_text_field parameter.

        Parameters
        ----------
        data
            Record to be embedded.

        Returns
        -------
        field
            Either a TextField or a ListField containing the record.
            Returns None if `data` is not a str or a dict.
        """
        if not isinstance(data, (str, Iterable)):
            self._logger.warning(
                "Cannot process data example %s of type %s", data, type(data)
            )
            return None

        if isinstance(data, str):
            data = [data]

        if isinstance(data, dict):
            data = data.values()

        if self._sentence_segmenter:
            text = self._text_transforms(self._value_as_string(data))
            sentences: List[TextField] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits[: self._max_nr_of_sentences]:
                word_tokens = self._tokenizer.tokenize(
                    sentence[: self._max_sequence_length]
                )
                sentences.append(TextField(word_tokens, self._token_indexers))
            return ListField(sentences) if sentences else None
        elif self._as_text_field:
            text = self._text_transforms(self._value_as_string(data))[
                : self._max_sequence_length
            ]
            word_tokens = self._tokenizer.tokenize(text)
            return TextField(word_tokens, self._token_indexers)

        # text_fields of different lengths are allowed, they will be sorted by the trainer and padded adequately
        text_fields = [
            TextField(
                self._tokenizer.tokenize(text[: self._max_sequence_length]),
                self._token_indexers,
            )
            for text in map(self._text_transforms, map(self._value_as_string, data))
            if text and text.strip()
        ]
        return ListField(text_fields) if text_fields else None

    # pylint: disable=arguments-differ
    def text_to_instance(self, **inputs) -> Instance:
        """ Convert an input text data into a allennlp Instance"""
        raise NotImplementedError
