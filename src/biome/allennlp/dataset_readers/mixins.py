from typing import Optional, Union, Any, Dict

from allennlp.data import Tokenizer, TokenIndexer
from allennlp.data.fields import ListField, TextField
from allennlp.data.tokenizers import WordTokenizer


class CacheableMixin(object):
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


class TextFieldBuilderMixin(object):
    """
        This ``TextFieldBuilderMixin`` build ``Fields`` for inputs in classification problems

        Parameters
        ----------

        tokenizer
            The allennlp ``Tokenizer`` for text tokenization in ``TextField``

        token_indexers
            The allennlp ``TokenIndexer`` dictionary for ``TextField`` configuration

        as_text_field
            Flag indicating how to generate the ``Field``. If enabled, the output Field
            will a ``TextField`` with text concatenation, else the result field will be
            a ``ListField`` of ``TextField``, one per input data value
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        as_text_field: bool = False,
    ):
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers
        self._as_text_field = as_text_field

    @staticmethod
    def _value_as_string(value):
        # TODO evaluate field value type for stringfy properly
        return str(value)

    def build_textfield(
        self, data: Union[str, dict]
    ) -> Optional[Union[ListField, TextField]]:
        if not data:
            return None

        if self._as_text_field or isinstance(data, str):
            text = data
            if isinstance(text, dict):
                text = " ".join(data.values())

            return TextField(self._tokenizer.tokenize(text), self._token_indexers)

        text_fields = [
            TextField(
                self._tokenizer.tokenize(self._value_as_string(field_value)),
                self._token_indexers,
            )
            for field_name, field_value in data.items()
            if field_value
        ]

        return ListField(text_fields) if len(text_fields) > 0 else None
