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
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import allennlp
import numpy
import torch
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import Field
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data import Vocabulary
from allennlp.data.fields import ListField
from allennlp.data.fields import MetadataField
from allennlp.data.fields import SequenceLabelField
from allennlp.data.fields import TextField
from allennlp.models.archival import CONFIG_NAME

from . import vocabulary
from .backbone import ModelBackbone
from .configuration import PipelineConfiguration
from .errors import MissingArgumentError
from .errors import WrongValueError
from .helpers import split_signature_params_by_predicate
from .modules.heads import TaskHead
from .modules.heads import TaskOutput


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


class PipelineModel(allennlp.models.Model):
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
        The model output
    """

    PREDICTION_FILE_NAME = "predictions.json"

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

    def text_to_instance(self, **inputs) -> Optional[Instance]:
        """Applies the head featurize method"""
        try:
            return self._head.featurize(**inputs)
        except KeyError as error:
            # missing inputs
            raise MissingArgumentError(arg_name=error.args[0])

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
        input_dicts: Iterable[Dict[str, Any]],
        predictions: Iterable[Dict[str, Any]],
    ) -> None:
        """Log predictions to a file for a model analysis and feedback sessions.

        Parameters
        ----------
        input_dicts
            Input data to the model
        predictions
            Returned predictions from the model
        """
        for input_dict, prediction in zip(input_dicts, predictions):
            self._prediction_logger.info(
                json.dumps(dict(inputs=input_dict, prediction=sanitize(prediction)))
            )

    def predict(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        """Returns a prediction given some input data based on the current state of the model

        The accepted input is dynamically calculated and can be checked via the `self.inputs` attribute
        (`print(Pipeline.inputs)`)

        Returns
        -------
        predictions: `Dict[str, numpy.ndarray]`
            A dictionary containing the predictions and additional information
        """
        inputs = self._model_inputs_from_args(*args, **kwargs)

        return self.predict_batch([inputs])[0]

    def predict_batch(
        self, input_dicts: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, numpy.ndarray]]:
        """Returns predictions given some input data based on the current state of the model

        The predictions will be computed batch-wise, which is faster
        than calling `self.predict` for every single input data.

        Parameters
        ----------
        input_dicts
            The input data. The keys of the dicts must comply with the `self.inputs` attribute
        """
        _, predictions = self._get_instances_and_predictions(input_dicts)

        # Log predictions if the prediction logger was initialized
        if hasattr(self, "_prediction_logger"):
            self._log_predictions(input_dicts, predictions)

        return predictions

    def _get_instances_and_predictions(
        self, input_dicts
    ) -> Tuple[List[Instance], List[Dict[str, numpy.ndarray]]]:
        """Returns instances from the input_dicts and their predictions.

        Helper method used by the predict and explain methods.

        Parameters
        ----------
        input_dicts
            The input data. The keys of the dicts must comply with the `self.inputs` attribute

        Returns
        -------
        (instances, predictions)
        """
        if self.training:
            warnings.warn(
                "Train model enabled. "
                "Disabling training mode automatically. You can manually disable it: "
                "self.eval() or self.train(False)"
            )
            self.eval()

        instances = [self.text_to_instance(**input_dict) for input_dict in input_dicts]
        try:
            predictions = self.forward_on_instances(instances)
        except Exception as error:
            raise WrongValueError(
                f"Failed to make predictions for '{input_dicts}'"
            ) from error

        return instances, predictions

    def explain(self, *args, n_steps: int, **kwargs) -> Dict[str, Any]:
        """Returns a prediction given some input data including the attribution of each token to the prediction.

        The attributions are calculated by means of the [Integrated Gradients](https://arxiv.org/abs/1703.01365) method.

        The accepted input is dynamically calculated and can be checked via the `self.inputs` attribute
        (`print(Pipeline.inputs)`)

        Parameters
        ----------
        n_steps: int
            The number of steps used when calculating the attribution of each token.
            If the number of steps is less than 1, the attributions will not be calculated.

        Returns
        -------
        predictions: `Dict[str, numpy.ndarray]`
            A dictionary containing the predictions and attributions
        """
        inputs = self._model_inputs_from_args(*args, **kwargs)

        return self.explain_batch([inputs], n_steps)[0]

    def explain_batch(self, input_dicts: Iterable[Dict[str, Any]], n_steps: int):
        """Returns a prediction given some input data including the attribution of each token to the prediction.

        The predictions will be computed batch-wise, which is faster
        than calling `self.predict` for every single input data.

        The attributions are calculated by means of the [Integrated Gradients](https://arxiv.org/abs/1703.01365) method.

        The accepted input is dynamically calculated and can be checked via the `self.inputs` attribute
        (`print(Pipeline.inputs)`)

        Parameters
        ----------
        input_dicts
            The input data. The keys of the dicts must comply with the `self.inputs` attribute
        n_steps
            The number of steps used when calculating the attribution of each token.
            If the number of steps is less than 1, the attributions will not be calculated.

        Returns
        -------
        predictions
            A list of dictionaries containing the predictions and attributions
        """
        instances, predictions = self._get_instances_and_predictions(input_dicts)

        explained_predictions = [
            self.__build_explained_prediction(prediction, instance, n_steps)
            for instance, prediction in zip(instances, predictions)
        ]

        return explained_predictions

    def __build_explained_prediction(
        self, prediction: Dict[str, numpy.array], instance: Instance, n_steps: int
    ):

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
            if isinstance(field, SequenceLabelField):
                return [
                    {"label": label, "token": token["token"]}
                    for label, token in zip(
                        field.labels, extract_field_tokens(field.sequence_field)
                    )
                ]
            if isinstance(field, MetadataField):
                return []
            raise WrongValueError(f"Cannot extract fields from [{type(field)}]")

        return {
            name: tokens
            for name, field in instance.items()
            for tokens in [extract_field_tokens(field)]
            if tokens
        }

    def _model_inputs_from_args(self, *args, **kwargs) -> Dict[str, Any]:
        """Returns model input data dictionary"""
        inputs = {k: v for k, v in zip(self.inputs, args)}
        inputs.update(kwargs)

        return inputs


allennlp.models.Model.register(PipelineModel.__name__, exist_ok=True)(PipelineModel)
