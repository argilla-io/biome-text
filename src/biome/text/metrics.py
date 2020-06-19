from typing import Dict

import torch
from allennlp.training.metrics import Metric


class MultiLabelF1Measure(Metric):
    """
    Computes overall F1 for multilabel classification tasks.
    Predictions sent to the __call__ function are logits and it turns them into 0 or 1s.
    Used for `classification heads` with the `multilabel` parameter enabled.
    """

    def __init__(self) -> None:
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0

    def __call__(
        self, predictions: torch.LongTensor, gold_labels: torch.LongTensor, **kwargs
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor with logits of shape (batch_size, ..., num_labels).
        gold_labels : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
            :param **kwargs:
        """
        # turn logits into one-hot predictions
        predictions = (predictions.data > 0.0).long()
        self._tp += (predictions * gold_labels).sum().item()
        self._fp += (predictions * (1 - gold_labels)).sum().item()
        self._fn += ((1 - predictions) * gold_labels).sum().item()

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        Parameters
        ----------
        reset
            If True, reset the metrics after getting them

        Returns
        -------
        metrics_dict
            A Dict with:
            - precision : `float`
            - recall : `float`
            - f1-measure : `float`
        """
        predicted_positives = self._tp + self._fp
        actual_positives = self._tp + self._fn

        precision = self._tp / predicted_positives if predicted_positives > 0 else 0
        recall = self._tp / actual_positives if actual_positives > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        if reset:
            self.reset()

        return {"precision": precision, "recall": recall, "fscore": f1}

    def reset(self):
        """Resets the metrics"""
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0
