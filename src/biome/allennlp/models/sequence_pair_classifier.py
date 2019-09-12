import logging
from inspect import signature
from typing import Dict, Optional

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
    FeedForward,
    TimeDistributed,
)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from overrides import overrides
from torch.nn import Linear, Dropout

from . import SequenceClassifier

logger = logging.getLogger(__name__)


@Model.register("sequence_pair_classifier")
class SequencePairClassifier(SequenceClassifier):
    N_INPUTS = 2
    """
    This ``SequencePairClassifier`` uses a siamese network architecture to perform a classification task between a pair
    of records or documents.

    The classifier can be configured to take into account the hierarchical structure of documents and multi-field records.

    A record/document can be (1) single-field (single sentence): composed of a sequence of
    tokens, or (2) multi-field (multi-sentence): a sequence of fields with each of the fields containing a sequence of
    tokens. In the case of multi-field a doc_seq2vec_encoder and optionally a doc_seq2seq_encoder should be configured,
    for encoding each of the fields into a single vector encoding the full record/doc must be configured.

    The sequences are encoded into two single vectors, the resulting vectors are concatenated and fed to a
    linear classification layer.

    Parameters
    ----------
    vocab
        A Vocabulary, required in order to compute sizes for input/output projections
        and passed on to the :class:`~allennlp.models.model.Model` class.
    text_field_embedder
        Used to embed the input text into a ``TextField``
    seq2seq_encoder
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    multifield_seq2vec_encoder
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `seq2vec_encoder`.
    multifield_seq2seq_encoder
        Optional Seq2Seq encoder layer for the encoded fields/sentences.
    dropout
        Dropout percentage to use on the output of the Seq2VecEncoder
    doc_dropout
        Dropout percentage to use on the output of the doc Seq2VecEncoder
    feed_forward
        A feed forward layer applied to the encoded inputs.
    initializer
        Used to initialize the model parameters.
    regularizer
        Used to regularize the model. Passed on to :class:`~allennlp.models.model.Model`.
    """

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
        combined_records = torch.cat(
            [self.forward_tokens(tokens) for tokens in [record1, record2]], dim=-1
        )

        return self.output_layer(combined_records, label)
