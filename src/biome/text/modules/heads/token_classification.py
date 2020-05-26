from typing import Dict, List, Optional, Union, cast

import torch
from allennlp.data import Instance, TextFieldTensors, Token
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.modules import ConditionalRandomField, FeedForward, TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

from biome.text.backbone import ModelBackbone
from biome.text.modules.specs import ComponentSpec, FeedForwardSpec
from biome.text import vocabulary
from .defs import TaskHead, TaskName, TaskOutput


class TokenClassification(TaskHead):
    """Task head for token classification (NER)"""

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        label_encoding: Optional[str] = "BIOUL",
        dropout: Optional[float] = None,
        feedforward: Optional[FeedForwardSpec] = None,
    ) -> None:
        super(TokenClassification, self).__init__(backbone)
        vocabulary.set_labels(self.backbone.vocab, labels)

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

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
        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": SpanBasedF1Measure(
                self.backbone.vocab,
                tag_namespace=vocabulary.LABELS_NAMESPACE,
                label_encoding=label_encoding,
            ),
        }
        # loss is calculated as -log_likelihood from crf
        self._loss = lambda logits, label, mask: -(self._crf(logits, label, mask))

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

    def forward(  # type: ignore
        self, text: TextFieldTensors, labels: torch.IntTensor = None
    ) -> TaskOutput:
        mask = get_text_field_mask(text)
        embedded_text = self.backbone.forward(text, mask)

        if self.dropout:
            encoded_text = self.dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._label_projection_layer(embedded_text)
        best_paths = self._crf.viterbi_tags(logits, mask)
        predicted_tags = cast(List[List[int]], [x[0] for x in best_paths])
        class_probabilities = logits * 0.0

        for i, instance_tags in enumerate(predicted_tags):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities[i, j, tag_id] = 1

        output = TaskOutput(
            logits=logits,
            probs=class_probabilities,
            mask=mask,
            tags=[
                [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
                for tags in predicted_tags
            ],
        )

        if labels is not None:
            output.loss = self._loss(logits, labels, mask)
            for metric in self._metrics.values():
                metric(class_probabilities, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self._metrics.items()
            if metric_name != "f1"
        }
        f1_dict = self._metrics.get("f1").get_metric(reset=reset)
        metrics.update(f1_dict)
        return metrics


class TokenClassificationSpec(ComponentSpec[TokenClassification]):
    """Spec for classification head components"""

    pass
