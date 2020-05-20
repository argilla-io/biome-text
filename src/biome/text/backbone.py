from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from allennlp.data import Instance, Token
from allennlp.data.fields import ListField, TextField
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from .featurizer import InputFeaturizer
from .modules.encoders import Encoder
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary


class ModelBackbone(torch.nn.Module):
    """The backbone of the model.

     It is composed of a tokenizer, featurizer and an encoder.
     This component of the model can be pretrained and used with different task heads.

     Parameters
     ----------
     vocab : `Vocabulary`
        The vocabulary of the pipeline
    tokenizer : `Tokenizer`
        Tokenizes the input depending on its type (str, List[str], Dict[str, Any])
    featurizer : `InputFeaturizer`
        Defines the input features of the tokens, indexes and embeds them.
    encoder : Encoder
        Outputs an encoded sequence of the tokens
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokenizer: Tokenizer,
        featurizer: InputFeaturizer,
        encoder: Optional[Encoder] = None,
    ):
        super(ModelBackbone, self).__init__()

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.encoder = (
            encoder.input_dim(self.embedder.get_output_dim()).compile()
            if encoder
            else PassThroughEncoder(self.embedder.get_output_dim())
        )

    @property
    def embedder(self) -> TextFieldEmbedder:
        return self.featurizer.embedder

    def forward(
        self,
        text: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        num_wrapping_dims: int = 0,
    ) -> torch.Tensor:
        """Applies embedding + encoder layers"""
        embeddings = self.embedder(text, num_wrapping_dims=num_wrapping_dims)
        return self.encoder(embeddings, mask=mask)

    def _update_vocab(self, vocab: Vocabulary, **kwargs):
        """This method is called when a base model updates the vocabulary"""
        self.vocab = vocab

        # This loop applies only for embedding layer.
        for model_path, module in self.embedder.named_modules():
            if hasattr(module, "extend_vocab"):
                module.extend_vocab(self.vocab)

    def featurize(
        self,
        record: Union[str, List[str], Dict[str, Any]],
        to_field: str = "record",
        aggregate: bool = False,
        tokenize: bool = True,
    ) -> Instance:
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
            return [
                key_tokens + value_tokens
                for key_tokens, value_tokens in self.tokenizer.tokenize_record(data).values()
            ]
        if isinstance(data, str):
            return [self.tokenizer.tokenize_text(data)]
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
                self.featurizer.indexer,
            )
        return ListField(
            [
                TextField(entry_tokens, self.featurizer.indexer)
                for entry_tokens in tokens
            ]
        )
