from typing import Any
from typing import Dict
from typing import List
from typing import Union

from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data import TokenIndexer
from allennlp.data.fields import ListField
from allennlp.data.fields import TextField

from biome.text.features import WordFeatures
from biome.text.tokenizer import Tokenizer


class InputFeaturizer:
    """Transforms input text (words and/or characters) into indexes and embedding vectors.

    This class defines two input features, words and chars for embeddings at word and character level respectively.

    You can provide additional features by manually specify `indexer` and `embedder` configurations within each
    input feature.

    Attributes
    ----------
    tokenizer : `Tokenizer`
        Tokenizes the input depending on its type (str, List[str], Dict[str, Any])
    indexer : `Dict[str, TokenIdexer]`
        Features dictionary for token indexing. Built from `FeaturesConfiguration`
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        indexer: Dict[str, TokenIndexer],
    ):
        self.tokenizer = tokenizer
        self.indexer = indexer

    def __call__(
        self,
        record: Union[str, List[str], Dict[str, Any]],
        to_field: str = "record",
        aggregate: bool = False,
        tokenize: bool = True,
        exclude_record_keys: bool = False,
    ) -> Instance:
        """Generates an allennlp instance from a record input.

        Parameters
        ----------
        record: `Union[str, List[str], Dict[str, Any]]`
            Input data
        to_field: `str`
            The field name in the returned instance
        aggregate: `bool`
            If true, the returned instance will contain a single `TextField` with all record fields;
            If false, the instance will contain a `ListField` of `TextField`s.
        tokenize: `bool`
            If false, skip tokenization phase and pass record data as tokenized token list.
        exclude_record_keys: `bool`
            If true, excludes record keys from the output tokens in the featurization of dictionaries

        Returns
        -------
        instance: `Instance`
        """
        data = record

        record_tokens = (
            self._data_tokens(data, exclude_record_keys)
            if tokenize
            else [[Token(t) for t in data]]
        )
        return Instance({to_field: self._tokens_to_field(record_tokens, aggregate)})

    def _data_tokens(
        self, data: Union[str, List[str], Dict[str, Any]], exclude_record_keys: bool
    ) -> List[List[Token]]:
        """Converts the input data into a list of list of tokens depending on its type.

        Parameters
        ----------
        data: Union[str, List[str], Dict[str, Any]]
            The input data
        exclude_record_keys: `bool`
            If true, excludes record keys from the output tokens in the featurization of dictionaries

        Returns
        -------
        list_of_list_of_tokens
        """
        if isinstance(data, dict):
            return self.tokenizer.tokenize_record(data, exclude_record_keys)
        if isinstance(data, str):
            return self.tokenizer.tokenize_text(data)
        return self.tokenizer.tokenize_document(data)

    def _tokens_to_field(
        self, tokens: List[List[Token]], aggregate: bool
    ) -> Union[ListField, TextField]:
        """Wraps the tokens with the indexer in one or several `TextField`s

        Parameters
        ----------
        tokens: List[List[Token]]
            The list of list of input tokens
        aggregate: `bool`
            If true, the returned instance will contain a single `TextField` with all record fields;
            If false, the instance will contain a `ListField` of `TextField`s.

        Returns
        -------
        field
            `TextField` if aggregate is true, `ListField` otherwise
        """
        if aggregate:
            return TextField(
                [token for entry_tokens in tokens for token in entry_tokens],
                self.indexer,
            )
        return ListField(
            [TextField(entry_tokens, self.indexer) for entry_tokens in tokens]
        )

    @property
    def has_word_features(self) -> bool:
        """Checks if word features are already configured as part of the featurization"""
        return WordFeatures.namespace in self.indexer
