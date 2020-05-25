import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.data import Instance, Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from .featurizer import InputFeaturizer
from .modules.encoders import Encoder


class ModelBackbone(torch.nn.Module):
    """The backbone of the model.

    It is composed of a tokenizer, featurizer and an encoder.
    This component of the model can be pretrained and used with different task heads.

    Attributes
    ----------
    vocab : `Vocabulary`
        The vocabulary of the pipeline
    featurizer : `InputFeaturizer`
        Defines the input features of the tokens and indexes
    embedder: `TextFieldEmbedder`
        The backbone embedder layer
    encoder : Encoder
        Outputs an encoded sequence of the tokens
    """

    def __init__(
        self,
        vocab: Vocabulary,
        featurizer: InputFeaturizer,
        embedder: TextFieldEmbedder,
        encoder: Optional[Encoder] = None,
    ):
        super(ModelBackbone, self).__init__()

        self.vocab = vocab
        self.featurizer = featurizer
        self.embedder = embedder
        self.encoder = (
            encoder.input_dim(self.embedder.get_output_dim()).compile()
            if encoder
            else PassThroughEncoder(self.embedder.get_output_dim())
        )

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
        for model_path, module in self.named_modules():
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

        Deprecated: use self.featurizer instead

        """
        warnings.warn(
            "backbone.featurize is deprecated. Use instead backbone.featurizer",
            DeprecationWarning,
        )
        return self.featurizer(record, to_field, aggregate, tokenize)
