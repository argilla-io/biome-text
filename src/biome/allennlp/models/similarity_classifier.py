from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
    FeedForward,
)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from torch.nn import CosineEmbeddingLoss
from torch.nn.modules import Module

from . import SequenceClassifier


@Model.register("similarity_classifier")
class SimilarityClassifier(SequenceClassifier):
    """
    This ``SimilarityClassifier`` uses a siamese network architecture to perform a binary classification task:
    are two inputs similar or not?
    The two input sequences are encoded with two single vectors, the resulting vectors are concatenated and fed to a
    linear classification layer.

    Apart from the CrossEntropy loss, you can add an additional loss term by setting the "verification" parameter to
    true. This will drive the network to create vector clusters for each "class" in the data.

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
    feed_forward
        A feed forward layer applied to the encoded inputs.
    verification
        Include a term in the loss function that rewards similar encoded vectors for similar inputs.
        Make sure that the label "same" is indexed as 0, and the label "different" as 1.
        Also, when using this loss term, make sure the dropout of the FeedForward layer is set to 0 in the last layer.
        (Deep Learning Face Representation by Joint Identification-Verification, https://arxiv.org/pdf/1406.4773.pdf)
    initializer
        Used to initialize the model parameters.
    regularizer
        Used to regularize the model. Passed on to :class:`~allennlp.models.model.Model`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feed_forward: Optional[FeedForward] = None,
        dropout: float = None,
        verification: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(SequenceClassifier, self).__init__(
            vocab, regularizer
        )  # Passing on kwargs does not work because of the 'from_params' machinery

        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._feed_forward = feed_forward

        if self._feed_forward:
            self._classifier_input_dim = self._feed_forward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        # Due to the concatenation of the two input vectors
        self._classifier_input_dim *= 2

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._num_labels
        )

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._loss_sim = CosineEmbeddingLoss(margin=0.5) if verification else None
        # The value 0.5 for the margin is a recommended conservative value, see:
        # https://pytorch.org/docs/stable/nn.html#cosineembeddingloss

        self._metrics = {
            label: F1Measure(index)
            for index, label in self.vocab.get_index_to_token_vocabulary(
                "labels"
            ).items()
        }

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        record1: Dict[str, torch.LongTensor],
        record2: Dict[str, torch.LongTensor],
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        The architecture is basically:
        Embedding -> Seq2Seq -> Seq2Vec -> Dropout -> FeedForward -> Concatenation -> Classification layer


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
        label
            A torch tensor indicating if the two set of input tokens are similar or not (dim: ``(batch_size, 2)``).

        Returns
        -------
        An output dictionary consisting of:
        logits
        class_probabilities
        loss : :class:`~torch.Tensor`, optional
            A scalar loss to be optimised.
        """
        embedded_texts = []
        for tokens in [record1, record2]:
            embedded_text = self._text_field_embedder(tokens)
            mask = get_text_field_mask(tokens).float()

            if self._seq2seq_encoder:
                embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

            embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

            if self._dropout:
                embedded_text = self._dropout(embedded_text)

            if self._feed_forward:
                embedded_text = self._feed_forward(embedded_text)

            embedded_texts.append(embedded_text)

        combined_records = torch.cat(embedded_texts, dim=-1)

        logits = self._classification_layer(combined_records)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "class_probabilities": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))

            if self._loss_sim:
                # we need to transform the labels, see:
                # https://pytorch.org/docs/stable/nn.html#cosineembeddingloss
                label_transformed = torch.where(
                    label == 0, torch.ones_like(label), -1 * torch.ones_like(label)
                )
                loss += self._loss_sim(
                    embedded_texts[0], embedded_texts[1], label_transformed.float()
                )

            output_dict["loss"] = loss
            self._accuracy(logits, label)
            for name, metric in self._metrics.items():
                metric(logits, label)

        """
        This was an idea of a similarity classifier, solely based on the distance of the vectors
        if self._distance:
            diff = embedded_texts[0] - embedded_texts[1]
            distance = torch.norm(diff, dim=-1, keepdim=True)
            m = 0.5 # This value should be optimized during training!!
            # The idea is to make a quick scan and choose the value that maximizes the accuracy!

            # This part should be replaced by a calibrated probability based on the distance
            logits = torch.cat([(distance < m).float(), (distance >= m).float()], dim=1)
            probs = torch.nn.functional.softmax(logits, dim=-1)

            output_dict = {
                "logits": logits,
                "class_probabilities": probs,
                "distance": distance,
            }

            if label is not None:
                loss = self._loss(distance.view(-1), label.float().view(-1), m)  # using the ContrastiveLoss()
                output_dict["loss"] = loss
                self._accuracy(logits, label)
                for name, metric in self._metrics.items():
                    metric(logits, label)
        """

        return output_dict


class ContrastiveLoss(Module):
    """Computes a contrastive loss given a distance.

    We do not use it at the moment, i leave it here just in case.
    """
    def forward(self, distance, label, margin):
        """Compute the loss.

        Important: Make sure label = 0 corresponds to the same case, label = 1 to the different case!

        Parameters
        ----------
        distance
            Distance between the two input vectors
        label
            Label if the two input vectors belong to the same or different class.
        margin
            If the distance is larger than the margin, the distance of different class vectors
            does not contribute to the loss.

        Returns
        -------
        loss

        """
        loss_same = (1 - label) * distance ** 2
        loss_diff = (
            label * torch.max(torch.zeros_like(distance), margin - distance) ** 2
        )

        loss = loss_same.sum() + loss_diff.sum()

        return loss
