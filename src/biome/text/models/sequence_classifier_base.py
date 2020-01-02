from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    TextFieldEmbedder,
    Seq2VecEncoder,
    Seq2SeqEncoder,
    FeedForward,
    TimeDistributed,
)
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn import Dropout, Linear
from .mixins import BiomeClassifierMixin


class SequenceClassifierBase(BiomeClassifierMixin, Model):
    """In the most simple form this ``BaseModelClassifier`` encodes a sequence with a ``Seq2VecEncoder``, then
    predicts a label for the sequence.

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
    dropout
        Dropout percentage to use on the output of the Seq2VecEncoder
    multifield_seq2seq_encoder
        Optional Seq2Seq encoder layer for the encoded fields.
    multifield_seq2vec_encoder
        If we use `ListField`s, this Seq2Vec encoder is required.
        If `multifield_seq2seq_encoder` is provided, this encoder will pool its output.
        Otherwise, this encoder will operate directly on the output of the `seq2vec_encoder`.
    multifield_dropout
        Dropout percentage to use on the output of the doc Seq2VecEncoder
    feed_forward
        A feed forward layer applied to the encoded inputs.
    initializer
        Used to initialize the model parameters.
    regularizer
        Used to regularize the model. Passed on to :class:`~allennlp.models.model.Model`.
    accuracy
        The accuracy you want to use. By default, we choose a categorical top-1 accuracy.
    """

    @property
    def n_inputs(self):
        """ This value is used for calculate the output layer dimension. Default value is 1"""
        return 1

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        multifield_seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        multifield_seq2vec_encoder: Optional[Seq2VecEncoder] = None,
        feed_forward: Optional[FeedForward] = None,
        dropout: Optional[float] = None,
        multifield_dropout: Optional[float] = None,
        initializer: Optional[InitializerApplicator] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        accuracy: Optional[CategoricalAccuracy] = None,
    ) -> None:
        # Passing on kwargs does not work because of the 'from_params' machinery
        super().__init__(accuracy=accuracy, vocab=vocab, regularizer=regularizer)

        self._initializer = initializer or InitializerApplicator()
        # embedding
        self._text_field_embedder = text_field_embedder
        # dropout for encoded vector
        self._dropout = Dropout(dropout) if dropout else None
        # loss function for training
        self._loss = torch.nn.CrossEntropyLoss()
        # default value for wrapping dimensions for masking = 0 (single field)
        self._num_wrapping_dims = 0

        if multifield_seq2vec_encoder:
            # 1. setup num_wrapping_dims to 1
            self._num_wrapping_dims = 1
            # 2. Wrap the seq2vec and seq2seq encoders in TimeDistributed to account for the extra dimension num_fields
            self._seq2seq_encoder = (
                TimeDistributed(seq2seq_encoder) if seq2seq_encoder else None
            )
            self._seq2vec_encoder = TimeDistributed(seq2vec_encoder)
            # 3. (Optionally) setup multifield_seq2seq_encoder
            self._multifield_seq2seq_encoder = multifield_seq2seq_encoder
            # 4. setup multifield_seq2vec_encoder
            self._multifield_seq2vec_encoder = multifield_seq2vec_encoder
            # doc vector dropout
            self._multifield_dropout = (
                Dropout(multifield_dropout) if multifield_dropout else None
            )
        else:
            # token sequence level encoders
            self._seq2seq_encoder = seq2seq_encoder
            self._seq2vec_encoder = seq2vec_encoder

            self._multifield_seq2vec_encoder = None
            self._multifield_seq2seq_encoder = None
            self._multifield_dropout = None

        # (Optional) Apply a feed forward before the classification layer
        self._feed_forward = feed_forward
        if self._feed_forward:
            self._classifier_input_dim = self._feed_forward.get_output_dim()
        elif self._multifield_seq2vec_encoder:
            self._classifier_input_dim = (
                self._multifield_seq2vec_encoder.get_output_dim()
            )
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        # Due to the concatenation of the n_inputs input vectors
        self._classifier_input_dim *= self.n_inputs
        self._output_layer = Linear(self._classifier_input_dim, self.num_classes)

        self._initializer(self)

    @property
    def num_classes(self):
        return self.vocab.get_vocab_size(namespace="labels")

    def forward_tokens(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply the whole forward chain but last layer (output)

        Parameters
        ----------
        tokens
            The tokens tensor

        Returns
        -------
        A ``Tensor``
        """
        # TODO: This will probably not work for single field input, we need to check the shape of record 1 and 2.
        mask = get_text_field_mask(
            tokens, num_wrapping_dims=self._num_wrapping_dims
        ).float()

        embedded_text = self._text_field_embedder(
            tokens, mask=mask, num_wrapping_dims=self._num_wrapping_dims
        )

        # seq2seq encoding for each token in field
        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        # seq2vec encoding for tokens --> field vector
        encoded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            encoded_text = self._dropout(encoded_text)

        if self._multifield_seq2vec_encoder:
            # For calculating the mask, we assume that all seq2vec encoders produce a null vector if they only receive
            # masked input.
            # The dict key does not matter in this case, you can give an arbitrary name, like 'encoded_text'
            multifield_mask = get_text_field_mask({"encoded_text": encoded_text})

            # seq2seq encoding for each field vector
            if self._multifield_seq2seq_encoder:
                encoded_text = self._multifield_seq2seq_encoder(encoded_text, mask=multifield_mask)

            # seq2vec encoding for field --> record vector
            encoded_text = self._multifield_seq2vec_encoder(encoded_text, mask=multifield_mask)

            if self._multifield_dropout:
                encoded_text = self._multifield_dropout(encoded_text)

        if self._feed_forward:
            encoded_text = self._feed_forward(encoded_text)

        return encoded_text

    def output_layer(
        self, encoded_text: torch.Tensor, label
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        An output dictionary consisting of:
        logits : :class:`~torch.Tensor`
            A tensor of shape ``(batch_size, num_classes)`` representing
            the logits of the classifier model.
        class_probabilities : :class:`~torch.Tensor`
            A tensor of shape ``(batch_size, num_classes)`` representing
            the softmax probabilities of the classes.
        loss : :class:`~torch.Tensor`, optional
            A scalar loss to be optimised."""
        # get logits and probs
        logits = self._output_layer(encoded_text)
        class_probabilities = torch.softmax(logits, dim=1)
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if label is not None:
            loss = self._loss(logits, label.long())
            output_dict["loss"] = loss
            for metric in self._biome_classifier_metrics.values():
                metric(logits, label)

        return output_dict

    @overrides
    def forward(
        self, *inputs
    ) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        raise NotImplementedError
