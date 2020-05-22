from typing import Any, Dict, List, Union

from allennlp.data import Instance, Token, TokenIndexer
from allennlp.data.fields import ListField, TextField
from biome.text.tokenizer import Tokenizer
from biome.text.features import WordFeatures

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
        self, tokenizer: Tokenizer, indexer: Dict[str, TokenIndexer],
    ):
        self.tokenizer = tokenizer
        self.indexer = indexer

    @property
    def has_word_features(self) -> bool:
        """Checks if word features is already configured as part of featurization"""
        return WordFeatures.namespace in self.indexer

    def featurize(
        self,
        record: Union[str, List[str], Dict[str, Any]],
        to_field: str = "record",
        aggregate: bool = False,
        tokenize: bool = True,
    ) -> Instance:
        return self(record, to_field, aggregate, tokenize,)

    def __call__(
        self,
        record: Union[str, List[str], Dict[str, Any]],
        to_field: str = "record",
        aggregate: bool = False,
        tokenize: bool = True,
    ):

        """
        Generate a allennlp Instance from a record input.

        If aggregate flag is enabled, the resultant instance will contains a single TextField's
        with all record fields; otherwhise, a ListField of TextFields.

        Parameters
        ----------
        record: `Union[str, List[str], Dict[str, Any]]`
            input data
        to_field: `str`
            field name in returned instance
        aggregate: `bool`
            set data aggregation flag
        tokenize: `bool`
            If disabled, skip tokenization phase, and pass record data as tokenized token list.

        Returns
        -------

        instance: `Instance`

        """
        # TODO: Allow exclude record keys in data tokenization phase
        data = record

        record_tokens = (
            self._data_tokens(data) if tokenize else [[Token(t) for t in data]]
        )
        return Instance({to_field: self._tokens_to_field(record_tokens, aggregate)})

    def _data_tokens(self, data: Any) -> List[List[Token]]:
        """Convert data into a list of list of token depending on data type"""
        if isinstance(data, dict):
            return self.tokenizer.tokenize_record(data)
        if isinstance(data, str):
            return self.tokenizer.tokenize_text(data)
        return self.tokenizer.tokenize_document(data)

    def _tokens_to_field(
        self, tokens: List[List[Token]], aggregate: bool
    ) -> Union[ListField, TextField]:
        """
        If aggregate, generates a TextField including flatten token list. Otherwise,
        a ListField of TextField is returned.
        """
        if aggregate:
            return TextField(
                [token for entry_tokens in tokens for token in entry_tokens],
                self.indexer,
            )
        return ListField(
            [TextField(entry_tokens, self.indexer) for entry_tokens in tokens]
        )
