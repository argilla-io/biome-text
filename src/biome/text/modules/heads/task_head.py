from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy
import torch
from allennlp.common import Registrable
from allennlp.data import Instance
from allennlp.data import Token as AllenNLPToken
from allennlp.data.fields import ListField
from allennlp.data.fields import TextField
from spacy.tokens import Token as SpacyToken

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.heads.task_prediction import Attribution
from biome.text.modules.heads.task_prediction import TaskPrediction
from biome.text.modules.heads.task_prediction import Token

if TYPE_CHECKING:
    from biome.text.configuration import PredictionConfiguration


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
        # Ideally the child class should overwrite this and provide its proper empty TaskPrediction
        self._empty_prediction = TaskPrediction()

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

    @property
    def empty_prediction(self) -> TaskPrediction:
        """An empty task prediction that is returned when an unexpected error occurs during inference."""
        return self._empty_prediction

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

    def featurize(self, *args, **kwargs) -> Instance:
        """Converts incoming data into an Allennlp `Instance`, used for pyTorch tensors generation

        Returns
        -------
        instance

        Raises
        ------
        FeaturizeError
        """
        raise NotImplementedError

    def make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
        prediction_config: "PredictionConfiguration",
    ) -> TaskPrediction:
        """Transforms the forward output to a task output, only used for predictions.

        Parameters
        ----------
        single_forward_output
            A single (not batched) output from the head's forward method
        instance
            The instance underlying the prediction
        prediction_config
            Configurations for the prediction

        Returns
        -------
        task_prediction
            A task specific output for the prediction
        """
        prediction = self._make_task_prediction(single_forward_output, instance)

        if prediction_config.add_tokens:
            prediction.tokens = self._extract_tokens(instance)

        if prediction_config.add_attributions:
            prediction.attributions = self._compute_attributions(
                single_forward_output, instance, **prediction_config.attributions_kwargs
            )

        return prediction

    def _make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
    ) -> TaskPrediction:
        """Makes a basic task prediction.

        Must be implemented by the child class.

        Parameters
        ----------
        single_forward_output
            A single (not batched) output from the head's forward method
        instance
            The instance underlying the prediction

        Returns
        -------
        task_prediction
            A task specific output for the prediction
        """
        # One could implement a generic solution to just pass on the forward_output, but it would be slow and i
        # recommend thinking about what a prediction should return, it is very likely not the same as for the forward.
        # Possible solution:
        # Dynamically create a dataclass with necessary fields: C = dataclasses.make_dataclass(...)
        # Inherit from TaskPrediction: return type("...", (C, TaskPrediction, ), {})(**forward_output)
        raise NotImplementedError("Predictions are not implemented in this head")

    def _extract_tokens(self, instance: Instance) -> List[Union[Token, List[Token]]]:
        """Extracts the tokens from all TextFields in an instance.

        This is a generic implementation and you might have to overwrite it for your specific head.

        Parameters
        ----------
        instance
            The instance underlying the prediction

        Returns
        -------
        tokens
        """
        tokens: List[Union[Token, List[Token]]] = []

        for field_name, field in instance.items():
            if isinstance(field, TextField):
                tokens += self._extract_tokens_from_text_field(field, field_name)
            elif isinstance(field, ListField):
                for single_field in field:
                    if isinstance(single_field, TextField):
                        tokens.append(
                            self._extract_tokens_from_text_field(
                                single_field, field_name
                            )
                        )

        return tokens

    def _extract_tokens_from_text_field(
        self, field: TextField, name: str
    ) -> List[Token]:
        """Helper function for `self._extract_tokens`"""
        return [
            Token(
                text=token.text,
                start=token.idx,
                end=self._get_token_end(token),
                field=name,
            )
            for token in field
        ]

    @staticmethod
    def _get_token_end(token: Union[AllenNLPToken, SpacyToken]) -> Optional[int]:
        """Helper function for `self._extract_tokens_from_text_field`.
        While AllenNLP tokens have an idx_end, spacy Tokens do not.
        """
        try:
            return token.idx_end
        except AttributeError:
            return token.idx + len(token.text) if isinstance(token.idx, int) else None

    def _compute_attributions(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
        **kwargs
    ) -> List[Union[Attribution, List[Attribution]]]:
        """Tries to attribute the prediction to input features.

        Must be implemented by the child class.

        Parameters
        ----------
        single_forward_output
            A single (not batched) output from the head's forward method
        instance
            The instance underlying the prediction

        Returns
        -------
        attributions
        """
        raise NotImplementedError(
            "Attributing the prediction to the input is not implemented in this head"
        )


class TaskHeadConfiguration(ComponentConfiguration[TaskHead]):
    """Layer spec for TaskHead components"""

    pass
