from typing import Optional, Dict, List

import torch
from allennlp.data import Vocabulary
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
    TimeDistributed,
)
from allennlp.common.checks import check_dimensions_match
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from allennlp.models.model import Model
from torch.nn import Linear
from allennlp.nn import util

from .mixins import BiomeClassifierMixin


class MultifieldBiMpm(BiomeClassifierMixin, Model):
    """
    This ``Model`` implements the BiMPM model described in `Bilateral Multi-Perspective Matching
    for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.

    This implementation adds the feature of being compatible with multiple inputs for the two records.
    The matching will be done for all possible combinations between the two records, that is:
    (r1_1, r2_1), (r1_1, r2_2), ..., (r1_2, r2_1), (r1_2, r2_2), ...

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    matcher_word : ``BiMpmMatching``
        BiMPM matching on the output of word embeddings of premise and hypothesis.
    encoder : ``Seq2SeqEncoder``
        First encoder layer for the premise and hypothesis
    matcher_forward : ``BiMPMMatching``
        BiMPM matching for the forward output of first encoder layer
    matcher_backward : ``BiMPMMatching``
        BiMPM matching for the backward output of first encoder layer
    aggregator : ``Seq2VecEncoder``
        Aggregator of all BiMPM matching vectors
    classifier_feedforward : ``FeedForward``
        Fully connected layers for classification.
    dropout : ``float``, optional (default=0.1)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    accuracy
        The accuracy you want to use. By default, we choose a categorical top-1 accuracy.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        matcher_word: BiMpmMatching,
        encoder: Seq2SeqEncoder,
        matcher_forward: BiMpmMatching,
        matcher_backward: BiMpmMatching,
        aggregator: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        dropout: float = 0.1,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        accuracy: Optional[CategoricalAccuracy] = None,
    ):
        super().__init__(accuracy=accuracy, vocab=vocab, regularizer=regularizer)
        self.text_field_embedder = text_field_embedder

        self.matcher_word = matcher_word

        self.encoder = encoder
        self.td_encoder = TimeDistributed(self.encoder)
        self.matcher_forward = matcher_forward
        self.matcher_backward = matcher_backward

        self.aggregator = aggregator

        matching_dim = (
            matcher_word.get_output_dim()
            + self.matcher_forward.get_output_dim()
            + self.matcher_backward.get_output_dim()
        )

        check_dimensions_match(
            matching_dim,
            self.aggregator.get_input_dim(),
            "sum of dim of all matching layers",
            "aggregator input dim",
        )

        self.classifier_feedforward = classifier_feedforward

        check_dimensions_match(
            self.aggregator.get_output_dim() * 2,
            self.classifier_feedforward.get_input_dim(),
            "Twice the output dimension of the aggregator (record1 and record2 will be concatenated)",
            "classifier feedforward input dim",
        )

        self.classifier_output_dim = self.classifier_feedforward.get_output_dim()
        self.output_layer = Linear(
            self.classifier_output_dim, self.vocab.get_vocab_size(namespace="labels")
        )

        self.dropout = torch.nn.Dropout(dropout)

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        record1: Dict[str, torch.LongTensor],
        record2: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
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
        mask_record1 = util.get_text_field_mask(record1, num_wrapping_dims=1)
        mask_record2 = util.get_text_field_mask(record2, num_wrapping_dims=1)

        # embedding and encoding of record1
        embedded_record1 = self.dropout(
            self.text_field_embedder(record1, num_wrapping_dims=1)
        )
        encoded_record1 = self.dropout(
            self.td_encoder(embedded_record1, mask=mask_record1)
        )

        # embedding and encoding of record2
        embedded_record2 = self.dropout(
            self.text_field_embedder(record2, num_wrapping_dims=1)
        )
        encoded_record2 = self.dropout(
            self.td_encoder(embedded_record2, mask=mask_record2)
        )

        multifield_matching_vector_record1: List[torch.Tensor] = []
        multifield_matching_vector_record2: List[torch.Tensor] = []
        multifield_matching_mask_record1: List[torch.Tensor] = []
        multifield_matching_mask_record2: List[torch.Tensor] = []

        def add_matching_result(
            matcher, record1_values, record1_mask, record2_values, record2_mask, record1_list, record2_list
        ):
            # utility function to get matching result and add to the result list
            matching_result = matcher(
                record1_values, record1_mask, record2_values, record2_mask
            )
            record1_list.extend(matching_result[0])
            record2_list.extend(matching_result[1])

        # calculate matching vectors from word embedding and layer encoding
        half_hidden_size = self.encoder.get_output_dim() // 2

        for i in range(embedded_record1.shape[1]):
            for j in range(embedded_record2.shape[1]):
                matching_vector_record1: List[torch.Tensor] = []
                matching_vector_record2: List[torch.Tensor] = []

                add_matching_result(
                    self.matcher_word,
                    embedded_record1[:, i, :, :],
                    mask_record1[:, i, :],
                    embedded_record2[:, j, :, :],
                    mask_record2[:, j, :],
                    matching_vector_record1,
                    matching_vector_record2,
                )
                add_matching_result(
                    self.matcher_forward,
                    encoded_record1[:, i, :, :half_hidden_size],
                    mask_record1[:, i, :],
                    encoded_record2[:, j, :, :half_hidden_size],
                    mask_record2[:, j, :],
                    matching_vector_record1,
                    matching_vector_record2,
                )
                add_matching_result(
                    self.matcher_backward,
                    encoded_record1[:, i, :, half_hidden_size:],
                    mask_record1[:, i, :],
                    encoded_record2[:, j, :, half_hidden_size:],
                    mask_record2[:, j, :],
                    matching_vector_record1,
                    matching_vector_record2,
                )

                # concat the matching vectors
                matching_vector_cat_record1 = self.dropout(
                    torch.cat(matching_vector_record1, dim=2)
                )
                matching_vector_cat_record2 = self.dropout(
                    torch.cat(matching_vector_record2, dim=2)
                )

                multifield_matching_vector_record1.append(matching_vector_cat_record1)
                multifield_matching_vector_record2.append(matching_vector_cat_record2)
                multifield_matching_mask_record1.append(mask_record1[:, i, :])
                multifield_matching_mask_record2.append(mask_record2[:, j, :])

        # concat the multifield vectors and masks
        multifield_matching_vector_cat_record1 = torch.cat(multifield_matching_vector_record1, dim=1)
        multifield_matching_vector_cat_record2 = torch.cat(multifield_matching_vector_record2, dim=1)
        multifield_matching_mask_cat_record1 = torch.cat(multifield_matching_mask_record1, dim=1)
        multifield_matching_mask_cat_record2 = torch.cat(multifield_matching_mask_record2, dim=1)

        # aggregate the matching vectors
        aggregated_record1 = self.dropout(
            self.aggregator(multifield_matching_vector_cat_record1, multifield_matching_mask_cat_record1)
        )
        aggregated_record2 = self.dropout(
            self.aggregator(multifield_matching_vector_cat_record2, multifield_matching_mask_cat_record2)
        )

        # the final forward layer
        logits = self.classifier_feedforward(
            torch.cat([aggregated_record1, aggregated_record2], dim=-1)
        )
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "class_probabilities": probs}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self._biome_classifier_metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict
