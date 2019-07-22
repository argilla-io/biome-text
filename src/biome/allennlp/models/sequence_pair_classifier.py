import logging
from typing import Dict, Optional
from overrides import overrides

import torch
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from torch.nn.functional import softmax

from . import SequenceClassifier

logger = logging.getLogger(__name__)


@Model.register("sequence_pair_classifier")
class SequencePairClassifier(SequenceClassifier):

    @overrides
    def forward(self,  # type: ignore
                record1: Dict[str, torch.Tensor],
                record2: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        record1 : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        record2 : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        label : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class label of shape
            ``(batch_size, num_classes)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input_record1 = self.text_field_embedder(record1)
        embedded_text_input_record2 = self.text_field_embedder(record2)
        mask_record1 = get_text_field_mask(record1)
        mask_record2 = get_text_field_mask(record2)
        if self.pre_encoder:
            embedded_text_input_record1 = self.pre_encoder(embedded_text_input_record1)
            embedded_text_input_record2 = self.pre_encoder(embedded_text_input_record2)
        if self.encoder:
            encoded_record1 = self.encoder(embedded_text_input_record1, mask_record1)
            encoded_record2 = self.encoder(embedded_text_input_record2, mask_record2)

        aggregated_records = torch.cat([encoded_record1, encoded_record2], dim=-1)
        decoded_text = self.decoder(aggregated_records)

        logits = self.output_layer(decoded_text)
        class_probabilities = softmax(logits, dim=1)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if label is not None:
            loss = self._loss(logits, label.long())
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            for name, metric in self.metrics.items():
                metric(logits, label)
        return output_dict
