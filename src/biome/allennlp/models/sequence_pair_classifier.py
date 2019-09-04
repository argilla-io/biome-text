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
    def forward(
        self,  # type: ignore
        record1: Dict[str, torch.Tensor],
        record2: Dict[str, torch.Tensor],
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        record1
            The first input tokens.
            The dictionary is the output of a ``TextField.as_array()``. It gives names to the tensors created by
            the ``TokenIndexer``s.
            In its most basic form, using a ``SingleIdTokenIndexer``, the dictionary is composed of:
            ``{"tokens": Tensor(batch_size, num_tokens)}``.
            The keys of the dictionary are defined in the `model.yml` input.
            The dictionary is designed to be passed on directly to a ``TextFieldEmbedder``, that has a
            ``TokenEmbedder`` for each key in the dictionary (except you set `allow_unmatched_keys` in the
            ``TextFieldEmbedder`` to False) and knows how to combine different word/character representations into a
            single vector per token in your input.
        record2
            The second input tokens.
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
        encoded_texts = []
        for tokens in [record1, record2]:
            # TODO dynamic num_wrapping_dims calculation from tokens tensor shape
            mask = get_text_field_mask(tokens, num_wrapping_dims=1).float()
            embedded_text = self._text_field_embedder(tokens, mask=mask)

            if self._pre_encoder:
                embedded_text = self._pre_encoder(embedded_text)

            if self._seq2seq_encoder:
                embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

            encoded_text = self._encoder(embedded_text, mask=mask)

            # apply dropout to encoded vector
            if self._dropout:
                encoded_text = self._dropout(encoded_text)

            encoded_texts.append(encoded_text)

        aggregated_records = torch.cat(encoded_texts, dim=-1)

        decoded_text = self._decoder(aggregated_records)

        logits = self._output_layer(decoded_text)
        class_probabilities = softmax(logits, dim=1)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if label is not None:
            loss = self._loss(logits, label.long())
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            for name, metric in self._metrics.items():
                metric(logits, label)
        return output_dict
