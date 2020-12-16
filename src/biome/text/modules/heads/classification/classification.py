from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy
import torch
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.data.fields import MultiLabelField
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure

from biome.text import helpers
from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.metrics import MultiLabelF1Measure

from ..task_head import TaskHead
from ..task_head import TaskName
from ..task_head import TaskOutput


class ClassificationHead(TaskHead):
    """Base abstract class for classification problems"""

    task_name = TaskName.text_classification

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
        if self._multilabel:
            self.metrics = {"macro": MultiLabelF1Measure()}
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.metrics = {"accuracy": CategoricalAccuracy()}
            self.metrics.update(
                {
                    "micro": FBetaMeasure(average="micro"),
                    "macro": FBetaMeasure(average="macro"),
                    "per_label": FBetaMeasure(
                        labels=[i for i in range(0, len(labels))]
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
        # "if not label:" fails for ndarrays this is why we explicitly check None
        if label is None:
            return instance

        field = None
        # check if multilabel and if adequate type
        if self._multilabel and isinstance(label, (list, numpy.ndarray)):
            label = label.tolist() if isinstance(label, numpy.ndarray) else label
            field = MultiLabelField(label, label_namespace=vocabulary.LABELS_NAMESPACE)
        # check if not multilabel and adequate type + check for empty strings
        if not self._multilabel and isinstance(label, (str, int)) and label:
            field = LabelField(label, label_namespace=vocabulary.LABELS_NAMESPACE)
        if not field:
            # We have label info but we cannot build the label field --> discard the instance
            return None

        instance.add_field(to_field, field)
        return instance

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
            probabilities_batch = output.logits.sigmoid()
        else:
            probabilities_batch = torch.nn.functional.softmax(output.logits, dim=-1)

        output.labels, output.probabilities = [], []
        if self.num_labels > 0:
            output.labels, output.probabilities = zip(
                *[
                    self._get_labels_and_probabilities(probs)
                    for probs in probabilities_batch
                ]
            )

        del output.logits

        return output

    def _get_labels_and_probabilities(
        self, probabilities: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        """Returns the labels and probabilities sorted by the probability (descending)

        The list of the returned probabilities can be larger than the input probabilities,
        since we add all defined labels in the head.

        Parameters
        ----------
        probabilities
            Probabilities of the model's prediction for one instance

        Returns
        -------
        labels, probabilities
        """
        all_classes_probs = torch.zeros(
            self.num_labels,  # this can be >= probabilities.size()[0]
            device=probabilities.get_device()
            if probabilities.get_device() > -1
            else None,
        )
        all_classes_probs[: probabilities.size()[0]] = probabilities
        sorted_indexes_by_prob = torch.argsort(
            all_classes_probs, descending=True
        ).tolist()

        labels, probabilities = zip(
            *[
                (
                    vocabulary.label_for_index(self.backbone.vocab, idx),
                    float(all_classes_probs[idx]),
                )
                for idx in sorted_indexes_by_prob
            ]
        )

        return labels, probabilities

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
        final_metrics = {}
        if "accuracy" in self.metrics.keys():
            final_metrics.update(
                {"accuracy": self.metrics["accuracy"].get_metric(reset)}
            )

        for metric_name in ["micro", "macro"]:
            if metric_name in self.metrics.keys():
                for k, v in self.metrics[metric_name].get_metric(reset).items():
                    final_metrics.update({"{}/{}".format(metric_name, k): v})

        if "per_label" in self.metrics.keys():
            for k, values in self.metrics["per_label"].get_metric(reset).items():
                for i, v in enumerate(values):
                    label = vocabulary.label_for_index(self.backbone.vocab, i)
                    # sanitize label using same patterns as tensorboardX to avoid summary writer warnings
                    label = helpers.sanitize_metric_name(label)
                    final_metrics.update({"_{}/{}".format(k, label): v})

        return final_metrics

    def single_label_output(
        self, logits: torch.Tensor, label: Optional[torch.IntTensor] = None
    ) -> TaskOutput:
        output = TaskOutput(logits=logits)

        if label is not None:
            output.loss = self._loss(logits, label.long())
            for metric in self.metrics.values():
                metric(logits, label)

        return output

    def multi_label_output(
        self, logits: torch.Tensor, label: Optional[torch.IntTensor] = None
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
