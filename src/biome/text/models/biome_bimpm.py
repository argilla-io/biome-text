from typing import Optional, Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models.bimpm import BiMpm
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
)
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

from .mixins import BiomeClassifierMixin


class BiomeBiMpm(BiomeClassifierMixin, BiMpm):
    """
    This ``Model`` implements BiMPM model described in `Bilateral Multi-Perspective Matching
    for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.
    Also please refer to the `TensorFlow implementation <https://github.com/zhiguowang/BiMPM/>`_ and
    `PyTorch implementation <https://github.com/galsang/BIMPM-pytorch>`_.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    matcher_word : ``BiMpmMatching``
        BiMPM matching on the output of word embeddings of premise and hypothesis.
    encoder1 : ``Seq2SeqEncoder``
        First encoder layer for the premise and hypothesis
    matcher_forward1 : ``BiMPMMatching``
        BiMPM matching for the forward output of first encoder layer
    matcher_backward1 : ``BiMPMMatching``
        BiMPM matching for the backward output of first encoder layer
    encoder2 : ``Seq2SeqEncoder``
        Second encoder layer for the premise and hypothesis
    matcher_forward2 : ``BiMPMMatching``
        BiMPM matching for the forward output of second encoder layer
    matcher_backward2 : ``BiMPMMatching``
        BiMPM matching for the backward output of second encoder layer
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
        encoder1: Seq2SeqEncoder,
        matcher_forward1: BiMpmMatching,
        matcher_backward1: BiMpmMatching,
        encoder2: Seq2SeqEncoder,
        matcher_forward2: BiMpmMatching,
        matcher_backward2: BiMpmMatching,
        aggregator: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        dropout: float = 0.1,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        accuracy: Optional[CategoricalAccuracy] = None,
    ):
        super().__init__(
            accuracy=accuracy,
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            matcher_word=matcher_word,
            encoder1=encoder1,
            matcher_forward1=matcher_forward1,
            matcher_backward1=matcher_backward1,
            encoder2=encoder2,
            matcher_forward2=matcher_forward2,
            matcher_backward2=matcher_backward2,
            aggregator=aggregator,
            classifier_feedforward=classifier_feedforward,
            dropout=dropout,
            initializer=initializer,
            regularizer=regularizer,
        )

        self.metrics = self._biome_classifier_metrics

    @overrides
    def forward(
        self,  # type: ignore
        record1: Dict[str, torch.LongTensor],
        record2: Dict[str, torch.LongTensor],
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        output_dict = super().forward(premise=record1, hypothesis=record2, label=label)
        output_dict["class_probabilities"] = output_dict.pop("probs")

        return output_dict
