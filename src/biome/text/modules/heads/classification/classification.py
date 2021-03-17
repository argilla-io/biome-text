import logging
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

from biome.text import helpers
from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.featurizer import FeaturizeError
from biome.text.metrics import Metrics
from biome.text.modules.heads.task_head import TaskHead
from biome.text.modules.heads.task_head import TaskName
from biome.text.modules.heads.task_prediction import Attribution
from biome.text.modules.heads.task_prediction import TaskPrediction


class ClassificationHead(TaskHead):
    """Base abstract class for classification problems"""

    task_name = TaskName.text_classification
    _LOGGER = logging.getLogger(__name__)

    def __init__(
        self, backbone: ModelBackbone, labels: List[str], multilabel: bool = False
    ):
        super().__init__(backbone)
        vocabulary.set_labels(self.backbone.vocab, labels)

        # label related configurations
        self._multilabel = multilabel

        # metrics and loss
        if self._multilabel:
            self._loss = torch.nn.BCEWithLogitsLoss()
            self._metrics = Metrics(
                micro={"type": "fbeta_multi_label", "average": "micro"},
                macro={"type": "fbeta_multi_label", "average": "macro"},
                per_label={
                    "type": "fbeta_multi_label",
                    "labels": [i for i in range(len(labels))],
                },
            )
        else:
            self._loss = torch.nn.CrossEntropyLoss()
            self._metrics = Metrics(
                accuracy={"type": "categorical_accuracy"},
                micro={"type": "fbeta", "average": "micro"},
                macro={"type": "fbeta", "average": "macro"},
                per_label={"type": "fbeta", "labels": [i for i in range(len(labels))]},
            )

    def _add_label(
        self,
        instance: Instance,
        label: Union[List[str], List[int], str, int],
        to_field: str = "label",
    ) -> Instance:
        """Adds the label field for classification into the instance data

        Helper function for the child's `self.featurize` method.

        Parameters
        ----------
        instance
            Add a label field to this instance
        label
            The label data
        to_field
            Name space of the field

        Returns
        -------
        instance
            If `label` is not None, return `instance` with the a label field added.
            Otherwise return just the given `instance`.

        Raises
        ------
        FeaturizeError
            If the label is an empty string or does not match the type:
            - (str, int) for single label
            - (list, np.array) for multi label
        """
        # "if not label:" fails for ndarrays, this is why we explicitly check for None
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
            raise FeaturizeError(f"Cannot create label field for `label={label}`!")

        instance.add_field(to_field, field)

        return instance

    def _make_forward_output(
        self, logits: torch.Tensor, label: Optional[torch.IntTensor]
    ) -> Dict[str, Any]:
        """Returns a dict with the logits and optionally the loss

        Helper function for the child's `self.forward` method.
        """
        if label is not None:
            return {
                "loss": self._compute_metrics_and_return_loss(logits, label),
                "logits": logits,
            }

        return {"logits": logits}

    def _compute_metrics_and_return_loss(
        self, logits: torch.Tensor, label: torch.IntTensor
    ) -> float:
        """Helper function for the `self._make_forward_output` method."""
        for metric in self._metrics.get_dict(is_train=self.training).values():
            metric(logits, label)

        if self._multilabel:
            # casting long to float for BCELoss
            # see https://discuss.pytorch.org/t/nn-bcewithlogitsloss-cant-accept-one-hot-target/59980
            return self._loss(
                logits.view(-1, self.num_labels),
                label.view(-1, self.num_labels).type_as(logits),
            )

        return self._loss(logits, label.long())

    def _compute_labels_and_probabilities(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
    ) -> Tuple[List[str], List[float]]:
        """Computes the probabilities based on the logits and looks up the labels

        This is a helper function for the `self._make_task_prediction` of the children.

        Parameters
        ----------
        single_forward_output
            A single (not batched) output from the head's forward method

        Returns
        -------
        (labels, probabilities)
        """
        logits = torch.from_numpy(single_forward_output["logits"])

        if self._multilabel:
            probabilities = logits.sigmoid()
        else:
            probabilities = torch.nn.functional.softmax(logits, dim=0)

        labels, all_probabilities = (
            self._add_and_sort_labels_and_probabilities(probabilities)
            if self.num_labels > 0
            else ([], [])
        )

        return labels, all_probabilities

    def _add_and_sort_labels_and_probabilities(
        self, probabilities: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        """Returns the labels and probabilities sorted by the probability (descending)

        Helper function for the `self._compute_labels_and_probabilities` method. The list of the returned
        probabilities can be larger than the input probabilities, since we add all defined labels in the head.

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

        labels = [
            vocabulary.label_for_index(self.backbone.vocab, idx)
            for idx in sorted_indexes_by_prob
        ]
        probabilities = [
            float(all_classes_probs[idx]) for idx in sorted_indexes_by_prob
        ]

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
        metrics, final_metrics = self._metrics.get_dict(is_train=self.training), {}
        if "accuracy" in metrics.keys():
            final_metrics.update({"accuracy": metrics["accuracy"].get_metric(reset)})

        for metric_name in ["micro", "macro"]:
            if metric_name in metrics.keys():
                for k, v in metrics[metric_name].get_metric(reset).items():
                    final_metrics.update({"{}/{}".format(metric_name, k): v})

        if "per_label" in metrics.keys():
            for k, values in metrics["per_label"].get_metric(reset).items():
                for i, v in enumerate(values):
                    label = vocabulary.label_for_index(self.backbone.vocab, i)
                    # sanitize label using same patterns as tensorboardX to avoid summary writer warnings
                    label = helpers.sanitize_metric_name(label)
                    final_metrics.update({"_{}/{}".format(k, label): v})

        return final_metrics

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def featurize(self, *args, **kwargs) -> Instance:
        raise NotImplementedError

    def _make_task_prediction(
        self, single_forward_output: Dict[str, numpy.ndarray], instance: Instance
    ) -> TaskPrediction:
        raise NotImplementedError

    def _compute_attributions(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
        **kwargs,
    ) -> List[Union[Attribution, List[Attribution]]]:
        raise NotImplementedError
