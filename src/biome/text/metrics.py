import copy
from typing import Any
from typing import Dict

from allennlp.common import Params
from allennlp.training.metrics import Metric


class Metrics:
    """Stores two dictionaries of identical metrics, one for training and one for validation.

    Parameters
    ----------
    **kwargs
        The key defines the name of the metric, the value must be a dictionary that can be used to instantiate a
        child class of `allennlp.training.metrics.Metric` via its `from_params` method.

    Examples
    --------
    >>> from allennlp.training.metrics import Metric
    >>> metrics = Metrics(accuracy={"type": "categorical_accuracy"}, f1={"type": "fbeta"})
    >>> for metric in metrics.get_dict(is_train=False).values():
    ...     assert isinstance(metric, Metric)
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        self.training_metrics = {}
        self.validation_metrics = {}
        for name, metric_kwargs in kwargs.items():
            # We need a special logic for the vocabulary, we do not want to deep copy it,
            # and it cannot be used in Params
            vocab = metric_kwargs.pop("vocabulary", None)
            self.training_metrics[name] = Metric.from_params(
                Params(copy.deepcopy(metric_kwargs)),
                **{} if vocab is None else {"vocabulary": vocab}
            )
            self.validation_metrics[name] = Metric.from_params(
                Params(metric_kwargs), **{} if vocab is None else {"vocabulary": vocab}
            )

    def get_dict(self, is_train: bool = True) -> Dict[str, Metric]:
        if is_train:
            return self.validation_metrics
        return self.training_metrics
