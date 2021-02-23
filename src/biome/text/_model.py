import inspect
import json
import logging
import os
import pickle
import warnings
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import allennlp
import pytorch_lightning as pl
import torch
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.models.archival import CONFIG_NAME

from . import vocabulary
from .backbone import ModelBackbone
from .configuration import PipelineConfiguration
from .configuration import PredictionConfiguration
from .featurizer import FeaturizeError
from .helpers import split_signature_params_by_predicate
from .modules.heads import TaskHead
from .modules.heads import TaskPrediction


class _HashDict(dict):
    """
    Hashable dict implementation.
    BE AWARE! Since dicts are mutable, the hash can change!
    """

    def __hash__(self):
        return pickle.dumps(self).__hash__()


class _HashList(list):
    """
    Hashable list implementation.
    BE AWARE! Since lists are mutable, the hash can change!
    """

    def __hash__(self):
        return pickle.dumps(self).__hash__()


class PipelineModel(allennlp.models.Model, pl.LightningModule):
    """
    This class represents pipeline model implementation for connect biome.text concepts with
    allennlp implementation details

    This class manages the head + backbone encoder, keeping the allennlnlp Model lifecycle. This class
    should be hidden to api users.

    Parameters
    ----------
    name
        Name of the pipeline model
    head
        TaskHead of the pipeline model

    Attributes
    ----------
    name: str
        Name of the pipeline model
    head: TaskHead
        TaskHead of the pipeline model
    vocab: Vocabulary
        The vocabulary of the model, comes from allennlp.models.Model
    file_path: Optional[str]
        File path to a serialized version of this pipeline model
    inputs: List[str]
        The model inputs
    output: List[str]
        The model outputs (not prediction): Corresponding to the `TaskHead.featurize` optional arguments.
    """

    PREDICTION_FILE_NAME = "predictions.json"
    _LOGGER = logging.getLogger(__name__)

    def __init__(self, name: str, head: TaskHead):
        super().__init__(vocab=head.backbone.vocab)

        self.name = name
        self._head = None
        self.set_head(head)

        self.file_path: Optional[str] = None

    def _update_head_related_attributes(self):
        """Updates the inputs/outputs and default mapping attributes, calculated from model head"""
        required, optional = split_signature_params_by_predicate(
            self._head.featurize, lambda p: p.default == inspect.Parameter.empty
        )
        self._inputs = self._head.inputs() or [p.name for p in required]
        self._output = [p.name for p in optional if p.name not in self._inputs] or [
            None
        ]

    @classmethod
    def from_params(
        cls: Type["PipelineModel"],
        params: Params,
        vocab: Optional[Vocabulary] = None,
        **extras,
    ) -> "PipelineModel":
        """
        Load the model implementation from params. We build manually each component from config sections.

        The param keys matches exactly with keys in yaml configuration files

        Parameters
        ----------
        params
            The config key in these params is used to build the model components
        vocab
            The vocabulary for the model
        **extras
            Necessary for AllenNLP from_params machinery

        Returns
        -------
        pipeline_model
        """

        config = params.pop("config")
        if not isinstance(config, PipelineConfiguration):
            config = PipelineConfiguration.from_params(config)

        vocab = vocab or vocabulary.create_empty_vocabulary()
        tokenizer = config.build_tokenizer()
        featurizer = config.features.compile_featurizer(tokenizer)
        embedder = config.build_embedder(vocab)

        return cls(
            name=config.name,
            head=config.head.compile(
                backbone=ModelBackbone(
                    vocab,
                    featurizer=featurizer,
                    embedder=embedder,
                    encoder=config.encoder,
                )
            ),
        )

    @property
    def head(self) -> TaskHead:
        """Get the model head"""
        return self._head

    def set_head(self, head: TaskHead) -> None:
        """Set a head and update related model attributes"""
        self._head = head
        self._update_head_related_attributes()

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """The main forward method just wraps the head forward method"""
        return self._head(*args, **kwargs)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Fetch metrics defined in head layer"""
        return self._head.get_metrics(reset)

    def text_to_instance(self, **inputs) -> Optional[Instance]:
        """Applies the head featurize method"""
        try:
            return self._head.featurize(**inputs)
        except FeaturizeError as error:
            # we cannot featurize the input (empty strings, etc.)
            self._LOGGER.warning(error)
        except TypeError as error:
            # probably wrong input arguments for the head
            raise TypeError(
                f"Please check your input arguments, expected: {self.inputs}, actual: {inputs.keys()}"
            ) from error

    def extend_vocabulary(self, vocab: Vocabulary):
        """Extend the model's vocabulary with `vocab`

        Parameters
        ----------
        vocab
            The model's vocabulary will be extended with this one.
        """
        # self.vocab and self._head.backbone.vocab point to the same vocab!
        self.vocab.extend_from_vocab(vocab)

        # updates the embedding matrices
        self.extend_embedder_vocab()
        # updates head specific things
        self._head.on_vocab_update()

    @property
    def inputs(self) -> List[str]:
        """The model inputs. Corresponding to head.featurize required argument names"""
        return self._inputs

    @property
    def output(self) -> List[str]:
        """The model outputs (not prediction): Corresponding to head.featurize optional argument names."""
        return self._output

    def init_prediction_logger(
        self, output_dir: str, max_bytes: int = 20000000, backup_count: int = 20
    ):
        """Initialize the prediction logger.

        If initialized we will log all predictions to a file called *predictions.json* in the `output_folder`.

        Parameters
        ----------
        output_dir
            Path to the folder in which we create the *predictions.json* file.
        max_bytes
            Passed on to logging.handlers.RotatingFileHandler
        backup_count
            Passed on to logging.handlers.RotatingFileHandler

        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        predictions_logger = logging.getLogger(output_dir)
        predictions_logger.setLevel(logging.DEBUG)
        # This flag avoids logging messages to be propagated to the parent loggers
        predictions_logger.propagate = False
        file_handler = RotatingFileHandler(
            os.path.join(output_dir, self.PREDICTION_FILE_NAME),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(logging.INFO)
        predictions_logger.addHandler(file_handler)

        setattr(self, "_prediction_logger", predictions_logger)

    def init_prediction_cache(self, max_size: int) -> None:
        """Initialize a prediction cache using the functools.lru_cache decorator

        Parameters
        ----------
        max_size
            Save up to max_size most recent items.
        """

        if hasattr(self, "_predict_with_cache"):
            warnings.warn(
                "Prediction cache already initiated!", category=RuntimeWarning
            )
            return

        predict_with_cache = lru_cache(maxsize=max_size)(self.predict)

        def predict_wrapper(*args, **kwargs):
            def hashable_value(value) -> Any:
                if isinstance(value, dict):
                    return _HashDict(value)
                if isinstance(value, (list, tuple)):
                    return _HashList(value)
                return value

            return predict_with_cache(
                *[hashable_value(arg_value) for arg_value in args],
                **{
                    key: hashable_value(input_value)
                    for key, input_value in kwargs.items()
                },
            )

        self.__setattr__("predict", predict_wrapper)
        self.__setattr__("_predict_with_cache", predict_with_cache)

    def _log_predictions(
        self,
        batch: Iterable[Dict[str, Any]],
        predictions: Iterable[TaskPrediction],
    ) -> None:
        """Log predictions to a file for a model analysis and feedback sessions.

        Parameters
        ----------
        batch
            Input data to the model
        predictions
            Returned predictions from the model
        """
        for input_dict, prediction in zip(batch, predictions):
            self._prediction_logger.info(
                json.dumps(
                    dict(inputs=input_dict, prediction=sanitize(prediction.as_dict()))
                )
            )

    def predict(
        self,
        batch: List[Dict[str, Union[str, List[str], Dict[str, str]]]],
        prediction_config: PredictionConfiguration,
    ) -> List[Optional[TaskPrediction]]:
        """Returns predictions given some input data based on the current state of the model

        The keys of the input dicts in the batch must coincide with the `self.inputs` attribute.

        Parameters
        ----------
        batch
            A list of dictionaries that represents a batch of inputs.
            The dictionary keys must comply with the `self.inputs` attribute.
        prediction_config
            Contains configurations for the prediction
        """
        if self.training:
            warnings.warn(
                "Training mode enabled."
                "Disabling training mode automatically. You can manually disable it: "
                "self.eval() or self.train(False)"
            )
            self.eval()

        instances = [self.text_to_instance(**input_dict) for input_dict in batch]
        # Filter out None instances, that is when the head could not create an instance out of the input
        none_indices, not_none_instances = [], []
        for i, instance in enumerate(instances):
            if instance is None:
                none_indices.append(i)
            else:
                not_none_instances.append(instance)

        try:
            forward_outputs = self.forward_on_instances(not_none_instances)
        except Exception as error:
            input_examples = [
                example for i, example in enumerate(batch) if i not in none_indices
            ]
            self._LOGGER.exception(error)
            self._LOGGER.warning(
                f"Failed to make a forward pass for '{input_examples}'"
            )
            return [None] * len(batch)

        predictions = []
        for forward_output, instance in zip(forward_outputs, not_none_instances):
            try:
                predictions.append(
                    self.head.make_task_prediction(
                        forward_output, instance, prediction_config
                    )
                )
            except Exception as error:
                self._LOGGER.exception(error)
                self._LOGGER.warning(
                    f"Failed to make a task prediction for '{forward_output, instance}'"
                )
                predictions.append(None)

        # Add None for the none instances
        for index in none_indices:
            predictions.insert(index, None)

        # Log predictions if the prediction logger was initialized
        if hasattr(self, "_prediction_logger"):
            self._log_predictions(batch, predictions)

        return predictions


allennlp.models.Model.register(PipelineModel.__name__, exist_ok=True)(PipelineModel)
