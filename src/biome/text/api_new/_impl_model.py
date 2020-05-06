import base64
import inspect
import json
import logging
import os
import pickle
import re
import warnings
from copy import deepcopy
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Tuple

import numpy
import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import prepare_environment, prepare_global_logging, sanitize
from allennlp.data import Instance, Vocabulary, DatasetReader
from allennlp.models import Model as AllennlpModel
from allennlp.models.archival import CONFIG_NAME, archive_model
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from dask.dataframe import Series as DaskSeries

from biome.text.api_new import PipelineConfiguration
from biome.text.api_new.data import DataSource
from biome.text.api_new.errors import MissingArgumentError
from biome.text.api_new.helpers import split_signature_params_by_predicate
from biome.text.api_new.model import Model
from biome.text.api_new.modules.heads import TaskHead
from biome.text.api_new.vocabulary import vocabulary


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


class _DataSourceReader(DatasetReader):
    """Base Allennlp DataSource reader"""

    def __init__(self, data_keys: List[str]):
        super(_DataSourceReader, self).__init__(lazy=True)
        self._default_ds_mapping = {k: k for k in data_keys if k}

    __LOGGER = logging.getLogger(__name__)

    def _read(self, file_path: str) -> Iterable[Instance]:
        """An generator that yields `Instance`s that are fed to the model

        This method is implicitly called when training the model.
        The predictor uses the `self.text_to_instance_with_data_filter` method.

        Parameters
        ----------
        file_path
            Path to the configuration file (yml) of the data source.

        Yields
        ------
        instance
            An `Instance` that is fed to the model
        """
        data_source = DataSource.from_yaml(
            file_path, default_mapping=self._default_ds_mapping
        )
        self.__LOGGER.debug("Read data set from %s", file_path)
        dataframe = data_source.to_mapped_dataframe()
        instances: DaskSeries = dataframe.apply(
            lambda x: self.text_to_instance(**x.to_dict()),
            axis=1,
            meta=(None, "object"),
        )

        return (instance for _, instance in instances.iteritems() if instance)

    def text_to_instance(self, **inputs) -> Instance:
        """ Convert an input text data into a allennlp Instance"""
        raise NotImplementedError


class _BaseModelImpl(AllennlpModel, _DataSourceReader):
    """
    This class is an internal implementation for connect biome.text concepts with allennlp implementation details

    This class manage the internal head + model architecture, keeping the allennlnlp Model lifecycle. This class
    must be hidden to api users.
    """

    __logger = logging.getLogger(__name__)

    def __init__(self, name: str, head: TaskHead):

        AllennlpModel.__init__(self, head.model.vocab)

        self.name = name
        self.head = head

        required, optional = split_signature_params_by_predicate(
            self.head.featurize, lambda p: p.default == inspect.Parameter.empty
        )
        self._inputs = self.head.inputs() or [p.name for p in required]
        self._output = ([p.name for p in optional] or [None])[0]

        _DataSourceReader.__init__(
            self, data_keys=[k for k in self.inputs + [self.output]]
        )

    @classmethod
    def from_params(
        cls: Type["_BaseModelImpl"],
        params: Params,
        vocab: Optional[Vocabulary] = None,
        **extras,
    ) -> "_BaseModelImpl":
        """
        Load the internal model implementation from params. We build manually each component from config sections.

        The param keys matches exactly with keys in yaml configuration files
        """

        config = params.pop("config")
        if not isinstance(config, PipelineConfiguration):
            config = PipelineConfiguration.from_params(config)

        return cls(
            name=config.name,
            head=config.head.compile(
                model=Model(
                    vocab=vocab
                    or vocabulary.empty_vocab(featurizer=config.features.compile()),
                    tokenizer=config.tokenizer.compile(),
                    featurizer=config.features.compile(),
                    encoder=config.encoder,
                )
            ),
        )

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """The main forward method. Wraps the head forward method and converts the head output into a dictionary"""
        head_output = self.head.forward(*args, **kwargs)
        return self.head.process_output(head_output).as_dict()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Fetch metrics defined in head layer"""
        return self.head.get_metrics(reset)

    def text_to_instance(self, **inputs: Dict[str, Any]) -> Optional[Instance]:
        """Applies the head featurize method"""
        try:
            return self.head.featurize(**inputs)
        except KeyError as error:
            # missing inputs
            raise MissingArgumentError(arg_name=error.args[0])

    def update_vocab(self, vocab: Vocabulary):
        """Update the model vocabulary and re-launch all vocab updates methods"""
        self.vocab = vocab
        self.head.model._update_vocab(vocab)  # pylint: disable=protected-access
        self.head._update_vocab(vocab)  # pylint: disable=protected-access

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
        prediction = self.forward_on_instance(instance)

        return self.head.prediction_explain(prediction=prediction, instance=instance)

    def _model_inputs_from_args(self, *args, **kwargs) -> Dict[str, Any]:
        """Returns model input data dictionary"""
        inputs = {k: v for k, v in zip(self.inputs, args)}
        inputs.update(kwargs)

        return inputs

    def launch_experiment(
        self,
        params: Params,
        serialization_dir: str,
        extend_vocab: bool = False,
        batch_weight_key: str = "",
        embedding_sources_mapping: Dict[str, str] = None,
    ) -> "_BaseModelImpl":
        """Launch a local experiment for model training"""

        trainer = _BaseModelImplTrainer(
            self,
            params,
            serialization_dir,
            extend_vocab,
            batch_weight_key,
            embedding_sources_mapping,
        )

        model, _ = trainer.train()
        return model


class _BaseModelImplTrainer:
    """
    Default trainer for ``_BaseModelImpl``

    Arguments
    ----------
    model : ``_BaseModelImpl``
        The trainable model
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment
    serialization_dir : ``str``
        The directory in which to save results and logs.
    extend_vocab: ``bool``, optional (default=False)
        If ``True``, we use the new instances to extend your vocabulary.
    batch_weight_key : ``str``, optional (default="")
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    embedding_sources_mapping: ``Dict[str, str]``, optional (default=None)
        mapping from model paths to the pretrained embedding filepaths
        used during fine-tuning.
    """

    __LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        model: _BaseModelImpl,
        params: Params,
        serialization_dir: str,
        extend_vocab: bool = False,
        batch_weight_key: str = "",
        embedding_sources_mapping: Dict[str, str] = None,
    ):
        self._model = model
        self._params = params
        self._serialization_dir = serialization_dir
        self._extend_vocab = extend_vocab
        self._batch_weight_key = batch_weight_key
        self._embedding_sources_mapping = embedding_sources_mapping

        self._iterator = None
        self._trainer = None
        self._all_datasets = self.datasets_from_params()
        self._evaluate_on_test = self._params.pop_bool("evaluate_on_test", False)

        self._setup()

    def _setup(self):
        """Setup the trainer components and local resources"""
        from allennlp.data import DataIterator

        prepare_environment(self._params)
        if os.path.exists(self._serialization_dir) and os.listdir(
            self._serialization_dir
        ):
            self.__LOGGER.info(
                f"Serialization directory ({self._serialization_dir}) "
                f"already exists and is not empty."
            )

        os.makedirs(self._serialization_dir, exist_ok=True)
        prepare_global_logging(self._serialization_dir, file_friendly_logging=True)

        serialization_params = deepcopy(self._params).as_dict(quiet=True)
        with open(
            os.path.join(self._serialization_dir, CONFIG_NAME), "w"
        ) as param_file:
            json.dump(serialization_params, param_file, indent=4)

        self._params.pop("model", None)

        if self._extend_vocab:
            self.extend_vocab(self._all_datasets)

        vocab = self._model.vocab
        vocab.save_to_files(os.path.join(self._serialization_dir, "vocabulary"))

        self._iterator = DataIterator.from_params(self._params.pop("iterator"))
        self._iterator.index_with(vocab)

        trainer_params = self._params.pop("trainer")
        no_grad_regexes = trainer_params.pop(
            "no_grad", ()
        )  # This could be nice to have exposed
        for name, parameter in self._model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        # TODO: Customize trainer for better biome integration
        self._trainer = Trainer.from_params(
            model=self._model,
            serialization_dir=self._serialization_dir,
            iterator=self._iterator,
            train_data=self._all_datasets["train"],
            validation_data=self._all_datasets.get("validation"),
            params=trainer_params,
        )

    def datasets_from_params(self) -> Dict[str, Iterable[Instance]]:
        """
        Load all the datasets specified by the config.

        """

        self._params.pop("dataset_reader")
        self._params.pop("validation_dataset_reader", None)
        train_data_path = self._params.pop("train_data_path")

        self.__LOGGER.info("Reading training data from %s", train_data_path)
        datasets: Dict[str, Iterable[Instance]] = {
            "train": self._model.read(train_data_path)
        }

        validation_data_path = self._params.pop("validation_data_path", None)
        if validation_data_path is not None:
            self.__LOGGER.info("Reading validation data from %s", validation_data_path)
            datasets["validation"] = self._model.read(validation_data_path)

        test_data_path = self._params.pop("test_data_path", None)
        if test_data_path is not None:
            self.__LOGGER.info("Reading test data from %s", test_data_path)
            datasets["test"] = self._model.read(test_data_path)

        return datasets

    def extend_vocab(self, source_datasets: Dict[str, Iterable[Instance]]):
        """
        Extends the inner model vocabulary from source datasets

        Parameters
        ----------
        source_datasets: ``Dict[str, Iterable[Instance]]``
            The source datasets
        """

        datasets_for_vocab_creation = set(
            self._params.pop("datasets_for_vocab_creation", source_datasets)
        )

        for dataset in datasets_for_vocab_creation:
            if dataset not in source_datasets:
                raise ConfigurationError(
                    f"invalid 'dataset_for_vocab_creation' {dataset}"
                )

        self.__LOGGER.info(
            "Extending model vocabulary using %s data.",
            ", ".join(datasets_for_vocab_creation),
        )

        vocabulary_params = self._params.pop("vocabulary", {})
        self._model.vocab.extend_from_instances(
            vocabulary_params,
            (
                instance
                for key, dataset in source_datasets.items()
                for instance in dataset
                if key in datasets_for_vocab_creation
            ),
        )
        self._model.update_vocab(vocab=self._model.vocab)

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

        if test_data and not self._evaluate_on_test:
            self.__LOGGER.info(
                "To evaluate on the test set after training, pass the "
                "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
            )
            return {}

        # Evaluate
        self.__LOGGER.info("The model will be evaluated using the best epoch weights.")
        return evaluate(
            self._model,
            test_data,
            data_iterator=self._iterator,
            cuda_device=self._trainer._cuda_devices[
                0
            ],  # pylint: disable=protected-access
            batch_weight_key=self._batch_weight_key,
        )

    def train(self) -> Tuple[_BaseModelImpl, Dict[str, Any]]:
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
        archive_model(
            self._serialization_dir, files_to_archive=self._params.files_to_archive
        )
