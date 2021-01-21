from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy
import torch
from allennlp.common import Registrable
from allennlp.data import Instance

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.heads.task_output import TaskOutput


class TaskName(Enum):
    """The task name enum structure"""

    none = 0
    text_classification = 1
    token_classification = 2
    language_modelling = 3

    def __str__(self) -> str:
        return str(self.name)

    def as_string(self) -> str:
        return str(self)


class TaskHead(torch.nn.Module, Registrable):
    """Base task head class"""

    """The task head name"""
    task_name: TaskName

    def __init__(self, backbone: ModelBackbone):
        super(TaskHead, self).__init__()
        self.backbone = backbone

    def on_vocab_update(self):
        """
        Actions when vocab is updated. Rebuild here modules that initialization depends on some vocab metric

        At this point, the model.vocab is already updated, so it could be used for architecture update
        """
        pass

    @classmethod
    def register(cls, overrides: bool = False, **kwargs):
        """Enables the task head component for pipeline loading"""
        super(TaskHead, TaskHead).register(cls.__name__, exist_ok=overrides)(cls)

    @property
    def labels(self) -> List[str]:
        """The configured vocab labels"""
        return vocabulary.get_labels(self.backbone.vocab)

    @property
    def num_labels(self):
        """The number of vocab labels"""
        return len(self.labels)

    def extend_labels(self, labels: List[str]):
        """Extends the number of labels"""
        vocabulary.extend_labels(self.backbone.vocab, labels)

    def inputs(self) -> Optional[List[str]]:
        """
        The expected inputs names for data featuring. If no defined,
        will be automatically calculated from featurize signature
        """
        return None

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """The head's forward pass, it must include the backbone's `forward`.

        When trained, the returned dict has to have a 'loss' key pointing to a
        scalar `torch.Tensor` representing the loss to be optimized.
        When used for inference, it has to include everything to make the TaskOutput -> `self.make_task_output`.
        """
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Metrics dictionary for training task"""
        raise NotImplementedError

    def featurize(self, *args, **kwargs) -> Optional[Instance]:
        """Converts incoming data into an allennlp `Instance`, used for pyTorch tensors generation"""
        raise NotImplementedError

    def make_task_output(
        self, single_forward_output: Dict[str, numpy.ndarray]
    ) -> TaskOutput:
        """Transforms the forward output to a task output, only used for predictions.

        Parameters
        ----------
        single_forward_output
            A single (not batched) output from the head's forward method

        Returns
        -------
        task_output
            A task specific output for the prediction
        """
        # One could implement a generic solution to just forward the forward_output:
        # dataclass with necessary fields: C = dataclasses.make_dataclass(...)
        # inherit from TaskOutput: return type("...", (C, TaskOutput, ), {})(**forward_output)
        raise NotImplementedError

    def explain_prediction(
        self, prediction: Dict[str, numpy.array], instance: Instance, n_steps: int
    ) -> Dict[str, Any]:
        """
        Adds embedding explanations information to prediction output

        Parameters
        ----------
        prediction: `Dict[str,, numpy.array]`
            The result input predictions
        instance: `Instance`
            The featurized input instance
        n_steps: int
            The number of steps to find token level attributions

        Returns
        -------
            Prediction with explanation
        """
        return {**prediction, "explain": {}}


class TaskHeadConfiguration(ComponentConfiguration[TaskHead]):
    """Layer spec for TaskHead components"""

    pass
