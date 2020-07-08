from typing import Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
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
    vocab
        The vocabulary of the pipeline
    featurizer
        Defines the input features of the tokens and indexes
    embedder
        The embedding layer
    encoder
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
        self, text: TextFieldTensors, mask: torch.Tensor, num_wrapping_dims: int = 0,
    ) -> torch.Tensor:
        """Applies the embedding and encoding layer

        Parameters
        ----------
        text
            Output of the `batch.as_tensor_dict()` method, basically the indices of the indexed tokens
        mask
            A mask indicating which one of the tokens are padding tokens
        num_wrapping_dims
            0 if `text` is the output of a `TextField`, 1 if it is the output of a `ListField`

        Returns
        -------
        tensor
            Encoded representation of the input
        """
        embeddings = self.embedder(text, num_wrapping_dims=num_wrapping_dims)
        return self.encoder(embeddings, mask=mask)

    def on_vocab_update(self):
        """This method is called when a base model updates the vocabulary"""
        # This loop applies only for embedding layer.
        for model_path, module in self.named_modules():
            if hasattr(module, "extend_vocab"):
                module.extend_vocab(self.vocab)
