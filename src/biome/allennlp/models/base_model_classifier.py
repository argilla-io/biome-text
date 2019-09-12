from abc import ABCMeta
from typing import Dict, Optional

import numpy as np
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
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from torch.nn import Dropout, Linear


class BaseModelClassifier(Model, metaclass=ABCMeta):
    """This ``BaseModelClassifier`` simply encodes a sequence with a ``Seq2VecEncoder``, then
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
        multifield_seq2vec_encoder: Optional[Seq2VecEncoder] = None,
        multifield_seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        feed_forward: Optional[FeedForward] = None,
        dropout: Optional[float] = None,
        doc_dropout: Optional[float] = None,
        initializer: Optional[InitializerApplicator] = None,
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        # Passing on kwargs does not work because of the 'from_params' machinery
        super(BaseModelClassifier, self).__init__(vocab, regularizer)

        self._initializer = initializer or InitializerApplicator()
        # embedding
        self._text_field_embedder = text_field_embedder
        # dropout for encoded vector
        self._dropout = Dropout(dropout) if dropout else None
        # doc vector dropout
        self._doc_dropout = Dropout(doc_dropout) if dropout else None
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
        # default value for wrapping dimensions for masking = 0 (single field)
        self._num_wrapping_dims = 0

        if multifield_seq2vec_encoder:
            # 1. setup num_wrapping_dims to 1
            self._num_wrapping_dims = 1
            # 2. Wrap the seq2vec and seq2seq encoders in TimeDistributed to account for the extra dimension num_fields
            self._seq2vec_encoder = TimeDistributed(seq2vec_encoder)
            self._seq2seq_encoder = (
                TimeDistributed(seq2seq_encoder) if seq2seq_encoder else None
            )
            # 3. setup multifield_seq2vec_encoder
            self._multifield_seq2vec_encoder = multifield_seq2vec_encoder
            # 4. (Optionally) setup multifield_seq2seq_encoder
            self._multifield_seq2seq_encoder = (
                multifield_seq2seq_encoder if multifield_seq2seq_encoder else None
            )
        else:
            # token sequence level encoders
            self._seq2seq_encoder = seq2seq_encoder
            self._seq2vec_encoder = seq2vec_encoder
            self._multifield_seq2vec_encoder = None
            self._multifield_seq2seq_encoder = None

        self._feed_forward = feed_forward
        if self._feed_forward:
            self._classifier_input_dim = self._feed_forward.get_output_dim()
        else:
            self._classifier_input_dim = (
                self._multifield_seq2vec_encoder.get_output_dim()
                if self._multifield_seq2vec_encoder
                # TODO why not self._seq2vec_encoder.get_output_dim() ????
                else self._seq2vec_encoder.get_input_dim()
            )

        # Due to the concatenation of the n_inputs input vectors
        self._classifier_input_dim *= self.n_inputs
        self._output_layer = Linear(self._classifier_input_dim, self.num_classes)

        self._initializer(self)

    @property
    def num_classes(self):
        return self.vocab.get_vocab_size(namespace="labels")

    def forward_tokens(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:

        """
            Apply the whole forward chain but last layer (output)
        Parameters
        ----------
        tokens The tokens tensor

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

        # seq2seq encoding for each field vector
        # TODO: Does not work, we need to mask properly
        if self._multifield_seq2seq_encoder:
            encoded_text = self._multifield_seq2seq_encoder(encoded_text)

        if self._doc_dropout:
            encoded_text = self._doc_dropout(encoded_text)

        # seq2vec encoding for field --> record vector
        if self._multifield_seq2vec_encoder:
            encoded_text = self._multifield_seq2vec_encoder(encoded_text)

        if self._feed_forward:
            encoded_text = self._feed_forward(encoded_text)

        return encoded_text

    def output_layer(
        self, encoded_text: torch.Tensor, label
    ) -> Dict[str, torch.Tensor]:
        """Returns
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
            self._accuracy(logits, label)
            for name, metric in self._metrics.items():
                metric(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict["class_probabilities"]
        if not isinstance(all_predictions, np.ndarray):
            all_predictions = all_predictions.data.numpy()

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        for i, probs in enumerate(all_predictions):
            argmax_i = np.argmax(probs)
            label = self.vocab.get_token_from_index(argmax_i, namespace="labels")
            label_prob = 0.0

            output_map_probs.append({})
            for j, prob in enumerate(probs):
                label_key = self.vocab.get_token_from_index(j, namespace="labels")
                output_map_probs[i][label_key] = prob
                if label_key == label:
                    label_prob = prob

            max_classes.append(label)
            max_classes_prob.append(label_prob)

        return {
            "logits": output_dict.get("logits"),
            "classes": output_map_probs,
            "max_class": max_classes,
            "max_class_prob": max_classes_prob,
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Get the metrics of our classifier, see :func:`~allennlp_2.models.Model.get_metrics`.

        Parameters
        ----------
        reset
            Reset the metrics after obtaining them?

        Returns
        -------
        A dictionary with all metric names and values.
        """
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self._metrics.items():
            precision, recall, f1 = metric.get_metric(
                reset
            )  # pylint: disable=invalid-name
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            all_metrics[metric_name + "/f1"] = f1
            all_metrics[metric_name + "/precision"] = precision
            all_metrics[metric_name + "/recall"] = recall

        num_metrics = len(self._metrics)
        all_metrics["average/f1"] = total_f1 / num_metrics
        all_metrics["average/precision"] = total_precision / num_metrics
        all_metrics["average/recall"] = total_recall / num_metrics
        all_metrics["accuracy"] = self._accuracy.get_metric(reset)

        return all_metrics
