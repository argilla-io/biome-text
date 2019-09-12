import logging
from typing import Dict

import torch
from allennlp.models.model import Model
from overrides import overrides

from biome.allennlp.models.base_model_classifier import BaseModelClassifier

logger = logging.getLogger(__name__)


@Model.register("sequence_classifier")
class SequenceClassifier(BaseModelClassifier):
    @overrides
    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.Tensor],
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens
            The input tokens.
            The dictionary is the output of a ``TextField.as_array()``. It gives names to the tensors created by
            the ``TokenIndexer``s.
            In its most basic form, using a ``SingleIdTokenIndexer``, the dictionary is composed of:
            ``{"tokens": Tensor(batch_size, num_tokens)}``.
            The keys of the dictionary are defined in the `model.yml` input.
            The dictionary is designed to be passed on directly to a ``TextFieldEmbedder``, that has a
            ``TokenEmbedder`` for each key in the dictionary (except you set `allow_unmatched_keys` in the
            ``TextFieldEmbedder`` to False) and knows how to combine different word/character representations into a
            single vector per token in your input.

        label
            A torch tensor representing the sequence of integer gold class label of shape
            ``(batch_size, num_classes)``.
        """
        encoded_text = self.forward_tokens(tokens)
        return self.output_layer(encoded_text, label)
