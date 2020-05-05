from enum import Enum
from typing import Any, Dict, List, Optional

import numpy
import torch
from allennlp.common import Registrable
from allennlp.data import Instance, Vocabulary

from biome.text.api_new.model import Model
from biome.text.api_new.modules.specs import ComponentSpec
from biome.text.api_new.vocabulary import vocabulary


class TaskOutput:
    """
    Task output data class

    A task output will contains almost the logits and probs properties
    """

    def __init__(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        **extra_data
    ):
        self.logits = logits
        self.probs = probs
        self.loss = loss

        for k, v in extra_data.items():
            self.__setattr__(k, v)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Dict reprentation of task output"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


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

    def __init__(self, model: Model):
        super(TaskHead, self).__init__()
        self.model = model

    def extend_vocab(self, vocab: Vocabulary, **kwargs):
        """This method is automatically called when a vocab is updated"""
        if self.model.vocab != vocab:
            self.model.vocab = vocab
        self._on_vocab_update()

    def _on_vocab_update(self):
        """
        Actions when vocab is updated. Rebuild here modules that initialization depends on some vocab metric

        At this point, the model.vocab is already updated, so it could be used for architecture update
        """
        pass

    @classmethod
    def register(cls, overrides: bool = False, **kwargs):
        """Enables the task head component for pipeline loading"""
        super(TaskHead, TaskHead).register(cls.__name__, overrides)(cls)

    @property
    def labels(self) -> List[str]:
        """The configured vocab labels"""
        return vocabulary.get_labels(self.model.vocab)

    @property
    def num_labels(self):
        """The number of vocab labels"""
        return len(self.labels)

    def extend_labels(self, labels: List[str]):
        """Extends the number of labels"""
        vocabulary.extend_labels(self.model.vocab, labels)

    def task_name(self) -> TaskName:
        """The task head name"""
        raise NotImplementedError

    def inputs(self) -> Optional[List[str]]:
        """
        The expected inputs names for data featuring. If no defined,
        will be automatically calculated from featurize signature
        """
        return None

    def forward(self, *args: Any, **kwargs: Any) -> TaskOutput:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Metrics dictionary for training task"""
        raise NotImplementedError

    def featurize(self, *args, **kwargs) -> Optional[Instance]:
        """Converts incoming data into an allennlp `Instance`, used for pyTorch tensors generation"""
        raise NotImplementedError

    def process_output(self, output: TaskOutput) -> TaskOutput:
        """Build extra parameters over basic task output"""
        return output

    def prediction_explain(
        self, prediction: Dict[str, numpy.array], instance: Instance
    ) -> Dict[str, Any]:
        """Adds embedding explanations information to prediction output"""
        raise {**prediction, "explain": {}}


class TaskHeadSpec(ComponentSpec[TaskHead]):
    """Layer spec for TaskHead components"""

    pass
