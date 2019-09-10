import logging
from typing import Dict, Optional
from overrides import overrides

import torch

from allennlp.modules import (
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
    FeedForward,
    TimeDistributed,
)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from . import SequenceClassifier

from biome.allennlp.models.utils import compute_and_set_layer_input_dim

logger = logging.getLogger(__name__)

@Model.register("sequence_pair_classifier")
class SequencePairClassifier(SequenceClassifier):
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
    doc_seq2vec_encoder
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `seq2vec_encoder`.
    doc_seq2seq_encoder
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
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        doc_seq2vec_encoder: Seq2VecEncoder = None,
        doc_seq2seq_encoder: Seq2SeqEncoder = None,
        feed_forward: Optional[FeedForward] = None,
        dropout: float = None,
        doc_dropout: float = None,
        initializer: Optional[InitializerApplicator] = None,
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(SequenceClassifier, self).__init__(
            vocab, regularizer
        )  # Passing on kwargs does not work because of the 'from_params' machinery
        self._initializer = initializer or InitializerApplicator()

        self._text_field_embedder = text_field_embedder

        # token sequence level encoders
        self._seq2seq_encoder = seq2seq_encoder # can be None
        self._seq2vec_encoder = seq2vec_encoder

        # default value for wrapping dimensions for masking = 0 (single field)
        self._num_wrapping_dims = 0

        # doc level encoders
        if doc_seq2vec_encoder:
            # If we want to use multi-field input we need to:

            # 0. if char encoder in text field embedder we wrap it in TimeDistributed
            for embedder in self._text_field_embedder._token_embedders:
                if embedder == "token_characters":
                    token_characters_embedder = self._text_field_embedder._token_embedders[embedder]
                    token_characters_embedder._encoder = TimeDistributed(token_characters_embedder._encoder)
                    self._text_field_embedder._token_embedders[embedder] = token_characters_embedder

            # 1. setup num_wrapping_dims to 1
            self._num_wrapping_dims = 1

            # 2. Wrap the seq2vec and seq2seq encoders in TimeDistributed to account for the extra dimension num_fields
            self._seq2vec_encoder = TimeDistributed(seq2vec_encoder)
            self._seq2seq_encoder = TimeDistributed(seq2seq_encoder) if seq2seq_encoder else None

            # 3. setup doc_seq2vec_encoder
            self._doc_seq2vec_encoder = doc_seq2vec_encoder

            # 4. (Optionally) setup doc_seq2seq_encoder
            self._doc_seq2seq_encoder = doc_seq2seq_encoder if doc_seq2seq_encoder else None

        # token vector dropout
        self._dropout = torch.nn.Dropout(dropout) if dropout else None

        # doc vector dropout
        self._doc_dropout = torch.nn.Dropout(doc_dropout) if dropout else None

        self._feed_forward = feed_forward

        if self._feed_forward:
            self._classifier_input_dim = self._feed_forward.get_output_dim()
        else:
            self._classifier_input_dim = self._doc_seq2vec_encoder.get_output_dim() if self._doc_seq2vec_encoder else self._seq2vec_encoder.get_input_dim()

        # Due to the concatenation of the two input vectors
        self._classifier_input_dim *= 2

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._num_labels
        )

        # metrics
        self._accuracy = CategoricalAccuracy()
        self._metrics = {
            label: F1Measure(index)
            for index, label in self.vocab.get_index_to_token_vocabulary(
            "labels"
        ).items()
        }

        # loss function for training
        self._loss = torch.nn.CrossEntropyLoss()

        self._initializer(self)

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
            # TODO: This will probably not work for single field input, we need to check the shape of record 1 and 2.
            mask = get_text_field_mask(tokens, num_wrapping_dims=self._num_wrapping_dims).float()

            embedded_text = self._text_field_embedder(tokens, mask=mask)

            # seq2seq encoding for each token in field
            if self._seq2seq_encoder:
                embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

            # seq2vec encoding for tokens --> field vector
            encoded_text = self._seq2vec_encoder(embedded_text, mask=mask)

            if self._dropout:
                encoded_text = self._dropout(encoded_text)

            # seq2seq encoding for each field vector
            # TODO: Does not work, we need to mask properly
            if self._doc_seq2seq_encoder:
                encoded_text = self._doc_seq2seq_encoder(encoded_text)

            if self._doc_dropout:
                encoded_text = self._doc_dropout(encoded_text)

            # seq2vec encoding for field --> record vector
            if self._doc_seq2vec_encoder:
                encoded_text = self._doc_seq2vec_encoder(encoded_text)

            if self._feed_forward:
                encoded_text = self._feed_forward(encoded_text)

            encoded_texts.append(encoded_text)

        combined_records = torch.cat(encoded_texts, dim=-1)

        logits = self._classification_layer(combined_records)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "class_probabilities": probs}

        if label is not None:
            loss = self._loss(logits, label.long())
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            for name, metric in self._metrics.items():
                metric(logits, label)
        return output_dict
