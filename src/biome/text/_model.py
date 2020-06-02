import inspect
import json
import logging
import os
import pathlib
import pickle
import re
import warnings
from copy import deepcopy
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

import allennlp
import numpy
import torch
from allennlp.common import Params
from allennlp.common.util import prepare_environment, sanitize
from allennlp.data import DataLoader, DatasetReader, Field, Instance, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.models.archival import CONFIG_NAME, archive_model
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from dask.dataframe import Series as DaskSeries
from torch.utils.data.dataset import Dataset

from biome.text import vocabulary
from .backbone import ModelBackbone
from .configuration import PipelineConfiguration
from .data import DataSource
from .errors import MissingArgumentError, WrongValueError
from .features import WordFeatures
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


class PipelineModel(allennlp.models.Model, allennlp.data.DatasetReader):
    """
    This class represents pipeline model implementation for connect biome.text concepts with
    allennlp implementation details

    This class manage the head + backbone encoder, keeping the allennlnlp Model lifecycle. This class
    should be hidden to api users.
    """

    def __init__(self, name: str, head: TaskHead):
        allennlp.models.Model.__init__(self, head.backbone.vocab)
        allennlp.data.DatasetReader.__init__(self, lazy=True)

        self._head = None
        self.name = name
        self.set_head(head)

    def _update_head_related_attributes(self):
        """Updates the inputs/outputs and default mapping attributes, calculated from model head"""
        required, optional = split_signature_params_by_predicate(
            self._head.featurize, lambda p: p.default == inspect.Parameter.empty
        )
        self._inputs = self._head.inputs() or [p.name for p in required]
        self._output = ([p.name for p in optional] or [None])[0]
        self._default_ds_mapping = {k: k for k in self._inputs + [self._output] if k}

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
        tokenizer = config.tokenizer.compile()
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

    def explain(self, *args, **kwargs) -> Dict[str, Any]:
        """Applies an prediction including token attribution explanation"""
        inputs = self._model_inputs_from_args(*args, **kwargs)
        instance = self.text_to_instance(**inputs)
        tokenization = self._get_instance_tokenization(instance)

        prediction = self.forward_on_instance(instance)
        explain = self._head.explain_prediction(
            prediction=prediction, instance=instance
        )
        # TODO: change explain data model include both, tokenization and attributions information
        #  We must apply this change in sync with frontend team.
        #  The data model proposal could be as follow:
        #    {
        #        "tokenization": { "text": ["This", "is", "an", "example"] },
        #        "attributions": { "text": [{"token": "This", "attribution": 0.39},...]}
        #    }
        #   For now, we add tokenization as a new separate field
        explain.update({"tokenization": tokenization})

        return explain

    def _get_instance_tokenization(self, instance: Instance) -> Dict[str, Any]:
        """Gets the tokenization information to current instance"""

        def extract_field_tokens(field: Field) -> Union[List[str], List[List[str]]]:
            """Tries to extract tokens from field"""
            if isinstance(field, TextField):
                return [token.text for token in cast(TextField, field).tokens]
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

    def launch_experiment(
        self,
        params: Params,
        serialization_dir: str,
        batch_weight_key: str = "",
        embedding_sources_mapping: Dict[str, str] = None,
    ) -> "PipelineModel":
        """Launch a local experiment for model training"""

        trainer = PipelineModelTrainer(
            self,
            params,
            serialization_dir,
            batch_weight_key,
            embedding_sources_mapping,
        )

        model, _ = trainer.train()
        return model

    def _read(self, file_path: str) -> Iterable[Instance]:
        """A generator that yields `Instance`s that are fed to the model

        This method is implicitly called when training the model.

        Parameters
        ----------
        file_path
            Path to the configuration file (yml) of the data source.

        Yields
        ------
        instance
            An `Instance` that is fed to the model
        """
        data_source = DataSource.from_yaml(file_path)
        if not data_source.mapping:
            data_source.mapping = self._default_ds_mapping

        dataframe = data_source.to_mapped_dataframe()
        instances: DaskSeries = dataframe.apply(
            lambda x: self.text_to_instance(**x.to_dict()),
            axis=1,
            meta=(None, "object"),
        )

        return (instance for _, instance in instances.iteritems() if instance)


class PipelineModelTrainer:
    """
    Default trainer for ``PipelineModel``

    Arguments
    ----------
    model : ``PipelineModel``
        The trainable model
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment
    serialization_dir : ``str``
        The directory in which to save results and logs.
    batch_weight_key : ``str``, optional (default="")
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    embedding_sources_mapping: ``Dict[str, str]``, optional (default=None)
        mapping from model paths to the pretrained embedding filepaths
        used during fine-tuning.
    """

    __LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        model: PipelineModel,
        params: Params,
        serialization_dir: str,
        batch_weight_key: str = "",
        embedding_sources_mapping: Dict[str, str] = None,
    ):
        self._model = model
        # TODO use biome classes instead of common parameters
        self._params = params
        self._serialization_dir = serialization_dir
        self._batch_weight_key = batch_weight_key
        self._embedding_sources_mapping = embedding_sources_mapping
        self._batch_size = self._params.pop("iterator")["batch_size"]

        self._trainer = None
        self._all_datasets = self.datasets_from_params()

        self._setup()

    def _setup(self):
        """Setup the trainer components and local resources"""

        prepare_environment(self._params)
        if os.path.exists(self._serialization_dir) and os.listdir(
            self._serialization_dir
        ):
            self.__LOGGER.info(
                f"Serialization directory ({self._serialization_dir}) "
                f"already exists and is not empty."
            )

        os.makedirs(self._serialization_dir, exist_ok=True)

        # We don't need to load pretrained weights from saved models
        try:
            self._params["model"]["config"]["features"]["word"].weights_file = None
        # there is no word feature
        except AttributeError:
            pass

        serialization_params = sanitize(deepcopy(self._params).as_dict(quiet=True))
        with open(
            os.path.join(self._serialization_dir, CONFIG_NAME), "w"
        ) as param_file:
            json.dump(serialization_params, param_file, indent=4)

        self._params.pop("model", None)

        vocab = self._model.vocab
        vocab.save_to_files(os.path.join(self._serialization_dir, "vocabulary"))

        for dataset in self._all_datasets.values():
            if dataset and hasattr(dataset, "index_with"):
                dataset.index_with(vocab)

        trainer_params = self._params.pop("trainer")
        no_grad_regexes = trainer_params.pop(
            "no_grad", ()
        )  # This could be nice to have exposed
        for name, parameter in self._model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        train_data_loader = DataLoader(
            self._all_datasets["train"], batch_size=self._batch_size
        )
        validation_ds = self._all_datasets.get("validation")
        validation_data_loader = (
            DataLoader(validation_ds, batch_size=self._batch_size)
            if validation_ds
            else None
        )

        # TODO: Customize trainer for better biome integration
        self._trainer = Trainer.from_params(
            model=self._model,
            serialization_dir=self._serialization_dir,
            data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            params=trainer_params,
        )

    def datasets_from_params(self) -> Dict[str, Dataset]:
        """
        Load all the datasets specified by the config.

        """

        self._params.pop("dataset_reader")
        self._params.pop("validation_dataset_reader", None)
        train_data_path = self._params.pop("train_data_path")

        self.__LOGGER.info("Reading training data from %s", train_data_path)
        datasets: Dict[str, Dataset] = {"train": self._model.read(train_data_path)}

        validation_data_path = self._params.pop("validation_data_path", None)
        if validation_data_path is not None:
            self.__LOGGER.info("Reading validation data from %s", validation_data_path)
            datasets["validation"] = self._model.read(validation_data_path)

        test_data_path = self._params.pop("test_data_path", None)
        if test_data_path is not None:
            self.__LOGGER.info("Reading test data from %s", test_data_path)
            datasets["test"] = self._model.read(test_data_path)

        return datasets

    def test_evaluation(self) -> Dict[str, Any]:
        """
        Evaluates the model against the test dataset (if defined)

        Returns
        -------
        Test metrics information

        """
        test_data = self._all_datasets.get("test")
        if not test_data:
            return {}

        self.__LOGGER.info("The model will be evaluated using the best epoch weights.")
        return evaluate(
            self._model,
            data_loader=DataLoader(test_data, batch_size=self._batch_size),
            cuda_device=self._trainer.cuda_device,
            batch_weight_key=self._batch_weight_key,
        )

    def train(self) -> Tuple[PipelineModel, Dict[str, Any]]:
        """
        Train the inner model with given configuration on initialization

        Returns
        -------
        A tuple with trained model and related metrics information
        """

        from allennlp.models.model import _DEFAULT_WEIGHTS

        try:
            metrics = self._trainer.train()
        except KeyboardInterrupt:
            # if we have completed an epoch, try to create a model archive.
            if os.path.exists(os.path.join(self._serialization_dir, _DEFAULT_WEIGHTS)):
                logging.info(
                    "Fine-tuning interrupted by the user. Attempting to create "
                    "a model archive using the current best epoch weights."
                )
                self.save_best_model()
            raise

        for k, v in self.test_evaluation().items():
            metrics["test_" + k] = v

        self.save_best_model()

        with open(
            os.path.join(self._serialization_dir, "metrics.json"), "w"
        ) as metrics_file:
            metrics_json = json.dumps(metrics, indent=2)
            metrics_file.write(metrics_json)

        return self._model, metrics

    def save_best_model(self):
        """Packages the best model as tar.gz archive"""
        archive_model(self._serialization_dir)
