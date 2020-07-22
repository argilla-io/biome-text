from typing import Dict, List, Optional, Union, cast

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Instance, TextFieldTensors
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.modules import ConditionalRandomField, FeedForward, TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import (
    ComponentConfiguration,
    FeedForwardConfiguration,
)
from .task_head import TaskHead, TaskName, TaskOutput


class TokenClassification(TaskHead):
    """Task head for token classification (NER)"""

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        label_encoding: Optional[str] = "BIOUL",
        top_k: int = 1,
        dropout: Optional[float] = 0.0,
        feedforward: Optional[FeedForwardConfiguration] = None,
    ) -> None:
        super(TokenClassification, self).__init__(backbone)
        vocabulary.set_labels(self.backbone.vocab, labels)

        self.top_k = top_k
        self.dropout = torch.nn.Dropout(dropout)
        self._feedforward: FeedForward = (
            None
            if not feedforward
            else feedforward.input_dim(backbone.encoder.get_output_dim()).compile()
        )
        # output layers
        self._classifier_input_dim = (
            self._feedforward.get_output_dim()
            if self._feedforward
            else backbone.encoder.get_output_dim()
        )
        # we want this linear applied to each token in the sequence
        self._label_projection_layer = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        )
        constraints = allowed_transitions(
            label_encoding,
            vocabulary.get_index_to_labels_dictionary(self.backbone.vocab),
        )
        self._crf = ConditionalRandomField(
            self.num_labels, constraints, include_start_end_transitions=True
        )

        self.metrics = {"accuracy": CategoricalAccuracy()}
        if self.top_k:
            self.metrics.update(
                {f"accuracy_{self.top_k}": CategoricalAccuracy(top_k=self.top_k)}
            )
        self.f1_metric = SpanBasedF1Measure(
            self.backbone.vocab,
            tag_namespace=vocabulary.LABELS_NAMESPACE,
            label_encoding=label_encoding,
        )

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """loss is calculated as -log_likelihood from crf"""
        return -1 * self._crf(logits, labels, mask)

    def featurize(
        self, text: List[str], labels: Optional[Union[List[str], List[int]]] = None
    ) -> Optional[Instance]:

        instance = self.backbone.featurizer(
            text, to_field="text", tokenize=False, aggregate=True
        )

        if labels:
            instance.add_field(
                "labels",
                SequenceLabelField(
                    labels,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace=vocabulary.LABELS_NAMESPACE,
                ),
            )

        return instance

    def task_name(self) -> TaskName:
        return TaskName.token_classification

    def decode(self, output: TaskOutput) -> TaskOutput:
        output.tags = (
            [
                [
                    vocabulary.label_for_index(self.backbone.vocab, idx)
                    for idx in tags[0]
                ]
                for instance_k_tags in output.k_tags
                for tags in instance_k_tags
            ],
        )

        del output.k_tags
        return output

    def forward(  # type: ignore
        self, text: TextFieldTensors, labels: torch.IntTensor = None
    ) -> TaskOutput:
        mask = get_text_field_mask(text)
        embedded_text = self.dropout(self.backbone.forward(text, mask))

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._label_projection_layer(embedded_text)
        # batch_paths = self._crf.viterbi_tags(logits, mask)
        # We just keep the best path for every instance
        batch_paths = self._crf.viterbi_tags(logits, mask, top_k=self.top_k)
        predicted_tags = cast(List[List[int]], [paths[0][0] for paths in batch_paths])
        class_probabilities = logits * 0.0

        for i, instance_tags in enumerate(predicted_tags):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities[i, j, tag_id] = 1

        output = TaskOutput(
            logits=logits, probs=class_probabilities, k_tags=batch_paths, mask=mask
        )

        if labels is not None:
            output.loss = self._loss(logits, labels, mask)
            for metric in list(self.metrics.values()) + [self.f1_metric]:
                metric(class_probabilities, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
            if metric_name != "f1"
        }
        metrics.update(self.f1_metric.get_metric(reset=reset))
        return metrics


class TokenClassificationConfiguration(ComponentConfiguration[TokenClassification]):
    """Configuration for classification head components"""

    pass
