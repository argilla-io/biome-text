import logging
from typing import Dict, Optional, List

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
from overrides import overrides
from torch.nn import CosineEmbeddingLoss
from torch.nn.modules import Module

from . import BaseModelClassifier

logger = logging.getLogger(__name__)


@Model.register("similarity_classifier")
class SimilarityClassifier(BaseModelClassifier):
    """
    This ``SimilarityClassifier`` uses a siamese network architecture to perform a binary classification task:
    are two inputs similar or not?
    The two input sequences are encoded with two single vectors, the resulting vectors are concatenated and fed to a
    linear classification layer.

    Apart from the CrossEntropy loss, this model includes a CosineEmbedding loss
    (https://pytorch.org/docs/stable/nn.html#cosineembeddingloss) that will drive the network to create
    vector clusters for each "class" in the data.
    Make sure that the label "same" is indexed as 0, and the label "different" as 1!!!
    Make sure that the dropout of the last Seq2Vec or the last FeedForward layer is set to 0!!!
    (Deep Learning Face Representation by Joint Identification-Verification, https://arxiv.org/pdf/1406.4773.pdf)

    Parameters
    ----------
    kwargs
        See the `BaseModelClassifier` for a description of the parameters.
    margin
        This parameter is passed on to the CosineEmbedding loss. It provides a margin,
        at which dissimilar vectors are not driven further apart.
        Can be between -1 (always drive apart) and 1 (never drive apart).
    verification_weight
        Defines the weight of the verification loss in the final loss sum:
        loss = CrossEntropy + w * CosineEmbedding
    """

    @property
    def n_inputs(self):
        # We need overwrite the number of inputs since this model accepts two inputs
        return 2

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
        margin: float = 0.5,
        verification_weight: float = 2.0,

    ):
        super().__init__(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            seq2vec_encoder=seq2vec_encoder,
            seq2seq_encoder=seq2seq_encoder,
            multifield_seq2seq_encoder=multifield_seq2seq_encoder,
            multifield_seq2vec_encoder=multifield_seq2vec_encoder,
            feed_forward=feed_forward,
            dropout=dropout,
            multifield_dropout=multifield_dropout,
            initializer=initializer,
            regularizer=regularizer,
        )  # Passing on kwargs does not work because of the 'from_params' machinery
        logger.warning(
            "Make sure that the label 'same' is indexed as 0, and the label 'different' as 1."
        )
        logger.warning(
            "Make sure that the dropout of the last Seq2Vec or the last FeedForward layer is set to 0."
        )

        self._verification_weight = verification_weight
        self._loss_sim = CosineEmbeddingLoss(margin=margin)
        # The value 0.5 for the margin is a recommended conservative value, see:
        # https://pytorch.org/docs/stable/nn.html#cosineembeddingloss

    @overrides
    def forward(
        self,  # type: ignore
        record1: Dict[str, torch.Tensor],
        record2: Dict[str, torch.Tensor],
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        The architecture is basically:
        Embedding -> Seq2Seq -> Seq2Vec -> Dropout -> (Optional: MultiField stuff) -> FeedForward
        -> Concatenation -> Classification layer

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
        encoded_texts = [self.forward_tokens(tokens) for tokens in [record1, record2]]

        return self.output_layer(encoded_texts, label)

    @overrides
    def output_layer(
        self, encoded_texts: List[torch.Tensor], label
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
        combined_records = torch.cat(encoded_texts, dim=-1)
        logits = self._output_layer(combined_records)
        class_probabilities = torch.softmax(logits, dim=1)
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if label is not None:
            loss = self._loss(logits, label.long())

            if self._loss_sim:
                # we need to transform the labels, see:
                # https://pytorch.org/docs/stable/nn.html#cosineembeddingloss
                label_transformed = torch.where(
                    label == 0, torch.ones_like(label), -1 * torch.ones_like(label)
                )
                loss += self._verification_weight * self._loss_sim(
                    encoded_texts[0], encoded_texts[1], label_transformed.float()
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
