from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.data import Instance
from allennlp.data.fields import LabelField, MultiLabelField
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from biome.text.backbone import ModelBackbone
from biome.text.vocabulary import vocabulary
from ..defs import TaskHead, TaskName, TaskOutput


class ClassificationHead(TaskHead):
    """Base abstract class for classification problems"""

    def __init__(
        self, backbone: ModelBackbone, labels: List[str], multilabel: bool = False
    ):
        super(ClassificationHead, self).__init__(backbone)
        vocabulary.set_labels(self.backbone.vocab, labels)

        # label related configurations
        self._multilabel = multilabel
        self.calculate_output = (
            self.multi_label_output if self._multilabel else self.single_label_output
        )

        # metrics and loss
        self.metrics = {"accuracy": CategoricalAccuracy()}
        if self._multilabel:
            # TODO: for multilabel we want to calculate F1 per label and/or ROC-AUC
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            # metrics, some AllenNLP models use the names _accuracy or _metrics, so we have to be more specific.
            self.metrics.update(
                {label: F1Measure(index) for index, label in enumerate(self.labels)}
            )
            self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, *args: Any, **kwargs: Any) -> TaskOutput:
        raise NotImplementedError

    def featurize(self, *args, **kwargs) -> Optional[Instance]:
        raise NotImplementedError

    def add_label(
        self,
        instance: Instance,
        label: Union[List[str], List[int], str, int],
        to_field: str = "label",
    ) -> Optional[Instance]:
        """Includes the label field for classification into the instance data"""

        if not label:
            return instance

        field = None
        if isinstance(label, list) and self._multilabel:
            field = MultiLabelField(label, label_namespace=vocabulary.LABELS_NAMESPACE)
        if isinstance(label, (str, int)) and not self._multilabel:
            field = LabelField(label, label_namespace=vocabulary.LABELS_NAMESPACE)
        if (
            not field
        ):  # We have label info but we cannot build the label field --> discard the instance
            return None

        instance.add_field(to_field, field)
        return instance

    def task_name(self) -> TaskName:
        return TaskName.text_classification

    def process_output(self, output: TaskOutput) -> TaskOutput:
        def labels_with_probabilities(probabilities: torch.Tensor) -> Dict[str, float]:
            """
            Calculates the descendant sorted label + probs dictionary
            using all output classes (not only predicted)
            """
            all_classes_probs = torch.zeros(
                self.num_labels,
                device=probabilities.get_device()
                if torch.cuda.is_available()
                else None,
            )
            all_classes_probs[:probabilities.size()[0]] = probabilities
            sorted_indexes_by_prob = torch.argsort(
                all_classes_probs, descending=True
            ).tolist()
            return {
                vocabulary.label_for_index(self.backbone.vocab, idx): all_classes_probs[
                    idx
                ]
                for idx in sorted_indexes_by_prob
            }

        probs_batch = output.probs

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        if self.num_labels > 0:
            for probs in probs_batch:
                labels_with_prob = labels_with_probabilities(probs)
                output_map_probs.append(labels_with_prob)

                label, prob = list(labels_with_prob.items())[0]
                max_classes.append(label)
                max_classes_prob.append(prob)

        output.classes = output_map_probs

        output.max_class = max_classes
        output.max_class_prob = max_classes_prob

        output.label = max_classes
        output.prob = max_classes_prob

        return output

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

        # TODO: Refactor and simplify
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self.metrics.items():
            if metric_name == "accuracy":
                all_metrics["accuracy"] = metric.get_metric(reset)
            else:
                # pylint: disable=invalid-name
                precision, recall, f_1 = metric.get_metric(
                    reset
                )  # pylint: disable=invalid-name
                total_f1 += f_1
                total_precision += precision
                total_recall += recall
                all_metrics[metric_name + "/f1"] = f_1
                all_metrics[metric_name + "/precision"] = precision
                all_metrics[metric_name + "/recall"] = recall

        num_classes = self.num_labels
        all_metrics["average/f1"] = total_f1 / num_classes
        all_metrics["average/precision"] = total_precision / num_classes
        all_metrics["average/recall"] = total_recall / num_classes

        return all_metrics

    def single_label_output(
        self, logits: torch.Tensor, label: Optional[torch.IntTensor] = None,
    ) -> TaskOutput:
        output = TaskOutput(
            logits=logits, probs=torch.nn.functional.softmax(logits, dim=-1)
        )

        if label is not None:
            output.loss = self._loss(logits, label.long())
            for metric in self.metrics.values():
                metric(logits, label)

        return output

    def multi_label_output(
        self, logits: torch.Tensor, label: Optional[torch.IntTensor] = None,
    ) -> TaskOutput:
        output = TaskOutput(logits=logits, probs=logits.sigmoid())

        if label is not None:
            # casting long to float for BCELoss
            # see https://discuss.pytorch.org/t/nn-bcewithlogitsloss-cant-accept-one-hot-target/59980
            output.loss = self._loss(
                logits.view(-1, self.num_labels),
                label.view(-1, self.num_labels).type_as(logits),
            )
            for metric in self.metrics.values():
                metric(logits, label)

        return output
