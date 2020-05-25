from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.data import Instance
from allennlp.data.fields import LabelField, MultiLabelField
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, FBetaMeasure

from biome.text.backbone import ModelBackbone
from biome.text import vocabulary
from ..defs import TaskHead, TaskName, TaskOutput
from biome.text import helpers


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
            # TODO: for multi-label we want to calculate F1 per label and/or ROC-AUC
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.metrics.update(
                {
                    "micro": FBetaMeasure(average="micro"),
                    "macro": FBetaMeasure(average="macro"),
                    "per_label": FBetaMeasure(
                        labels=[i for i in range(0, len(labels) - 1)]
                    ),
                }
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

    def decode(self, output: TaskOutput) -> TaskOutput:
        """Completes the output for the prediction

        Mainly adds probabilities and keys for the UI.

        Parameters
        ----------
        output
            The output from the head's forward method

        Returns
        -------
        completed_output
        """
        if self._multilabel:
            probabilities = output.logits.sigmoid()
        else:
            probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
        output.probs = probabilities

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        if self.num_labels > 0:
            for probs in probabilities:
                labels_with_prob = self._labels_with_probabilities(probs)
                output_map_probs.append(labels_with_prob)

                label, prob = list(labels_with_prob.items())[0]
                max_classes.append(label)
                max_classes_prob.append(prob)

        output.classes = output_map_probs

        output.max_class = max_classes  # deprecated
        output.max_class_prob = max_classes_prob  # deprecated

        output.label = max_classes
        output.prob = max_classes_prob

        return output

    def _labels_with_probabilities(
        self, probabilities: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculates the descendant sorted label + probs dictionary
        using all output classes (not only predicted)
        """
        all_classes_probs = torch.zeros(
            self.num_labels,
            device=probabilities.get_device() if torch.cuda.is_available() else None,
        )
        all_classes_probs[: probabilities.size()[0]] = probabilities
        sorted_indexes_by_prob = torch.argsort(
            all_classes_probs, descending=True
        ).tolist()

        return {
            vocabulary.label_for_index(self.backbone.vocab, idx): all_classes_probs[idx]
            for idx in sorted_indexes_by_prob
        }

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
        metrics = {}
        for name, metric in self.metrics.items():
            if name == "accuracy":
                metrics["accuracy"] = metric.get_metric(reset)
            elif name in ["macro", "micro"]:
                metrics.update(
                    {
                        "{}/{}".format(name, k): v
                        for k, v in metric.get_metric(reset).items()
                    }
                )
            elif name == "per_label":
                for k, values in metric.get_metric(reset).items():
                    for i, v in enumerate(values):
                        label = vocabulary.label_for_index(self.backbone.vocab, i)
                        # sanitize label using same patters as tensorboardX to avoid warnings
                        label = helpers.clean_metric_name(label)
                        metrics.update({"{}/{}".format(k, label): v})
        return metrics

    def single_label_output(
        self, logits: torch.Tensor, label: Optional[torch.IntTensor] = None,
    ) -> TaskOutput:
        output = TaskOutput(logits=logits)

        if label is not None:
            output.loss = self._loss(logits, label.long())
            for metric in self.metrics.values():
                metric(logits, label)

        return output

    def multi_label_output(
        self, logits: torch.Tensor, label: Optional[torch.IntTensor] = None,
    ) -> TaskOutput:
        output = TaskOutput(logits=logits)

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
