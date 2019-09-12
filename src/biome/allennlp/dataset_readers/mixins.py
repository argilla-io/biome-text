from typing import Optional, Union, Any

from allennlp.data.fields import ListField, TextField
from allennlp.data.tokenizers import WordTokenizer


class CacheableMixin(object):
    _datasets_cache = dict()

    @staticmethod
    def get(key) -> Optional[Any]:
        return CacheableMixin._datasets_cache.get(key, None)

    @staticmethod
    def set(key, data):
        CacheableMixin._datasets_cache[key] = data


class TextFieldBuilderMixin(object):
    def __init__(self, tokenizer, token_indexers, as_text_field):
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
