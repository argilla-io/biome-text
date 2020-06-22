import inspect
import json
import logging
import os
import pathlib
import pickle
import warnings
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast

import allennlp
import numpy
import torch
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import DataLoader, Field, Instance, Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.models.archival import CONFIG_NAME
from dask.dataframe import Series as DaskSeries

from . import vocabulary
from .backbone import ModelBackbone
from .configuration import PipelineConfiguration
from .data import DataSource
from .errors import MissingArgumentError, WrongValueError
from .helpers import split_signature_params_by_predicate
from .modules.heads import TaskHead, TaskOutput


class _HashDict(dict):
    """
    Hashable dict implementation.
    BE AWARE! Since dicts are mutable, the hash can change!
    """

    def __hash__(self):
        # user a better way
        return pickle.dumps(self).__hash__()


class _HashList(list):
    """
    Hashable list implementation.
    BE AWARE! Since dicts are mutable, the hash can change!
    """

    def __hash__(self):
        return pickle.dumps(self).__hash__()


class PipelineModel(allennlp.models.Model):
    """
    This class represents pipeline model implementation for connect biome.text concepts with
    allennlp implementation details

    This class manage the head + backbone encoder, keeping the allennlnlp Model lifecycle. This class
    should be hidden to api users.
    """

    PREDICTION_FILE_NAME = "predictions.json"

    def __init__(self, name: str, head: TaskHead):
        allennlp.models.Model.__init__(self, head.backbone.vocab)

        self._head = None
        self.name = name
        self.set_head(head)

    def _update_head_related_attributes(self):
        """Updates the inputs/outputs and default mapping attributes, calculated from model head"""
        required, optional = split_signature_params_by_predicate(
            self._head.featurize, lambda p: p.default == inspect.Parameter.empty
        )
        self._inputs = self._head.inputs() or [p.name for p in required]
        self._output = (
            [p.name for p in optional if p.name not in self._inputs] or [None]
        )[0]

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
        """

        config = params.pop("config")
        if not isinstance(config, PipelineConfiguration):
            config = PipelineConfiguration.from_params(config)

        vocab = vocab or vocabulary.empty_vocabulary(namespaces=config.features.keys)
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

    def cache_data(self, cache_directory: str) -> None:
        """Sets the cache data directory"""
        self._cache_directory = pathlib.Path(cache_directory)
        os.makedirs(self._cache_directory, exist_ok=True)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """The main forward method. Wraps the head forward method and converts the head output into a dictionary"""
        head_output: TaskOutput = self._head.forward(*args, **kwargs)
        # we don't want to break AllenNLP API: TaskOutput -> as_dict()
        return head_output.as_dict()

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Completes the output for the prediction

        Parameters
        ----------
        output_dict
            The `TaskOutput` from the model's forward method as dict

        Returns
        -------
        output_dict
            Completed output
        """
        # we don't want to break AllenNLP API: dict -> TaskOutput -> dict
        output = TaskOutput(**output_dict)
        completed_output = self._head.decode(output)
        return completed_output.as_dict()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Fetch metrics defined in head layer"""
        return self._head.get_metrics(reset)

    def text_to_instance(self, **inputs: Dict[str, Any]) -> Optional[Instance]:
        """Applies the head featurize method"""
        try:
            return self._head.featurize(**inputs)
        except KeyError as error:
            # missing inputs
            raise MissingArgumentError(arg_name=error.args[0])

    def set_vocab(self, vocab: Vocabulary):
        """Replace the current vocab and reload all model layer"""
        self.vocab = vocab
        self._head.backbone.vocab = vocab
        self._head.backbone.on_vocab_update()
        self._head.on_vocab_update()

    @property
    def inputs(self) -> List[str]:
        """The model inputs. Corresponding to head.featurize required argument names"""
        return self._inputs

    @property
    def output(self) -> Optional[str]:
        """The model output (not prediction): Corresponding to head.featurize optional argument names. Should be one"""
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

    def log_prediction(
        self, inputs: Dict[str, Any], prediction: Dict[str, Any]
    ) -> None:
        """Store prediction for model analysis and feedback sessions"""
        if hasattr(self, "_prediction_logger"):
            self._prediction_logger.info(
                json.dumps(dict(inputs=inputs, annotation=sanitize(prediction)))
            )

    def predict(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        """Make a single model prediction"""
        if self.training:
            warnings.warn(
                "Train model enabled. "
                "Disabling training mode automatically. You can manually disable it: "
                "self.eval() or self.train(False)"
            )
            self.eval()

        inputs = self._model_inputs_from_args(*args, **kwargs)
        instance = self.text_to_instance(**inputs)
        prediction = self.forward_on_instance(instance)
        self.log_prediction(inputs, prediction)

        return prediction

    def explain(self, *args, n_steps: int, **kwargs) -> Dict[str, Any]:
        """
        Applies an prediction including token attribution explanation

        Parameters
        ----------
        n_steps: int
            The number of steps for token attribution calculation (if proceed).
            If the number of steps is less than 1, the attributions will not be calculated
        args and kwargs:
            Dynamic arguments aligned to the current model head input features.

        Returns
        -------
            The input prediction data include information about prediction explanation
        """
        inputs = self._model_inputs_from_args(*args, **kwargs)
        instance = self.text_to_instance(**inputs)
        prediction = self.forward_on_instance(instance)
        explained_prediction = (
            prediction
            if n_steps <= 0
            else self._head.explain_prediction(
                prediction=prediction, instance=instance, n_steps=n_steps
            )
        )
        # If no explain was found, we return input tokenization as default
        # TODO: We should use an explain data structure instead of dict
        if not explained_prediction.get("explain"):
            explained_prediction["explain"] = self._get_instance_tokenization(instance)
        return explained_prediction

    def _get_instance_tokenization(self, instance: Instance) -> Dict[str, Any]:
        """Gets the tokenization information to current instance"""

        def extract_field_tokens(
            field: Field,
        ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
            """Tries to extract tokens from field"""

            def token_2_attribution(
                token: Token, attribution: float = 0.0
            ) -> Dict[str, Any]:
                return {"token": token.text, "attribution": attribution}

            if isinstance(field, TextField):
                return [
                    token_2_attribution(token)
                    for token in cast(TextField, field).tokens
                ]
            if isinstance(field, ListField):
                return [
                    extract_field_tokens(inner_field)
                    for inner_field in cast(ListField, field)
                ]
            raise WrongValueError(f"Cannot extract fields from [{type(field)}]")

        return {name: extract_field_tokens(field) for name, field in instance.items()}

    def _model_inputs_from_args(self, *args, **kwargs) -> Dict[str, Any]:
        """Returns model input data dictionary"""
        inputs = {k: v for k, v in zip(self.inputs, args)}
        inputs.update(kwargs)

        return inputs
