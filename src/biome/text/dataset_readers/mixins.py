import logging
from typing import Optional, Union, Any, Dict, List, Iterable

from allennlp.data import Tokenizer, TokenIndexer
from allennlp.data.fields import ListField, TextField
from allennlp.data.tokenizers import WordTokenizer, SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


class CacheableMixin:
    """
        This ``CacheableMixin`` allow in memory cache mechanism
    """

    _cache = dict()

    @staticmethod
    def get(key) -> Optional[Any]:
        """ Get a value from cache by key """
        return CacheableMixin._cache.get(key, None)

    @staticmethod
    def set(key, data):
        """ Set an cache entry """
        CacheableMixin._cache[key] = data


# pylint: disable=too-few-public-methods
class TextFieldBuilderMixin:
    """
    This ``TextFieldBuilderMixin`` build ``Fields`` for inputs in classification problems

    Parameters
    ----------
    tokenizer
        The allennlp ``Tokenizer`` for text tokenization in ``TextField``
    token_indexers
        The allennlp ``TokenIndexer`` dictionary for ``TextField`` configuration
    segment_sentences
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
    as_text_field
        Flag indicating how to generate the ``Field``. If enabled, the output Field
        will be a ``TextField`` with text concatenation, else the result field will be
        a ``ListField`` of ``TextField``, one per input data value
    max_sequence_length
        If you want to truncate the text input to a maximum number of characters
    max_nr_of_sentences
        Use only the first max_nr_of_sentences when segmenting the text into sentences
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        segment_sentences: Union[bool, SentenceSplitter] = False,
        as_text_field: bool = True,
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
    ):
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers
        self._sentence_segmenter = segment_sentences
        if segment_sentences is True:
            self._sentence_segmenter = SpacySentenceSplitter()
        self._as_text_field = as_text_field
        self._max_sequence_length = max_sequence_length
        self._max_nr_of_sentences = max_nr_of_sentences

        self._logger = logging.getLogger(self.__class__.__name__)

        if segment_sentences and not as_text_field:
            self._logger.warning(
                "You cannot segment sentences and set as_text_field to false at the same time, "
                "i will set as_text_field for the single sentences to true."
            )

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
                "Processing data %s of type %s as string", data, type(data)
            )
            data = self._value_as_string(data)

        if isinstance(data, str):
            data = [data]

        if isinstance(data, dict):
            data = data.values()

        if self._sentence_segmenter:
            text = self._value_as_string(data)
            sentences: List[TextField] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits[: self._max_nr_of_sentences]:
                word_tokens = self._tokenizer.tokenize(
                    sentence[: self._max_sequence_length]
                )
                sentences.append(TextField(word_tokens, self._token_indexers))
            return ListField(sentences) if sentences else None
        elif self._as_text_field:
            text = self._value_as_string(data)[: self._max_sequence_length]
            word_tokens = self._tokenizer.tokenize(text)
            return TextField(word_tokens, self._token_indexers)

        # text_fields of different lengths are allowed, they will be sorted by the trainer and padded adequately
        text_fields = [
            TextField(
                self._tokenizer.tokenize(
                    self._value_as_string(value)[: self._max_sequence_length]
                ),
                self._token_indexers,
            )
            for value in data
        ]
        return ListField(text_fields) if text_fields else None
