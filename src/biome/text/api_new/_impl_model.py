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
from tempfile import mkdtemp
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
        file_friendly_logging: bool = False,
        batch_weight_key: str = "",
        embedding_sources_mapping: Dict[str, str] = None,
    ) -> "_BaseModelImpl":
        """
        Fine tunes the given model, using a set of parameters that is largely identical to those used
        for :func:`~allennlp.commands.train.train_model`, except that the ``model`` section is ignored,
        if it is present (as we are already given a ``Model`` here).

        The main difference between the logic done here and the logic done in ``train_model`` is that
        here we do not worry about vocabulary construction or creating the model object.  Everything
        else is the same.

        Parameters
        ----------
        params : ``Params``
            A parameter object specifying an AllenNLP Experiment
        serialization_dir : ``str``
            The directory in which to save results and logs.
        extend_vocab: ``bool``, optional (default=False)
            If ``True``, we use the new instances to extend your vocabulary.
        file_friendly_logging : ``bool``, optional (default=False)
            If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
            down tqdm's output to only once every 10 seconds.
        batch_weight_key : ``str``, optional (default="")
            If non-empty, name of metric used to weight the loss on a per-batch basis.
        embedding_sources_mapping: ``Dict[str, str]``, optional (default=None)
            mapping from model paths to the pretrained embedding filepaths
            used during fine-tuning.
        """

        from allennlp.data import DataIterator
        from allennlp.models.model import _DEFAULT_WEIGHTS

        logger = self.__logger

        def datasets_from_params(
            model: _BaseModelImpl, params: Params
        ) -> Dict[str, Iterable[Instance]]:
            """
            Load all the datasets specified by the config.

            Parameters
            ----------
            model : ``_BaseModelImpl``
            params : ``Params``
            """

            params.pop("dataset_reader")
            params.pop("validation_dataset_reader", None)
            train_data_path = params.pop("train_data_path")

            logger.info("Reading training data from %s", train_data_path)
            datasets: Dict[str, Iterable[Instance]] = {
                "train": model.read(train_data_path)
            }

            validation_data_path = params.pop("validation_data_path", None)
            if validation_data_path is not None:
                logger.info("Reading validation data from %s", validation_data_path)
                datasets["validation"] = model.read(validation_data_path)

            test_data_path = params.pop("test_data_path", None)
            if test_data_path is not None:
                logger.info("Reading test data from %s", test_data_path)
                datasets["test"] = model.read(test_data_path)

            return datasets

        prepare_environment(params)
        if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
            logger.info(
                f"Serialization directory ({serialization_dir}) "
                f"already exists and is not empty."
            )

        os.makedirs(serialization_dir, exist_ok=True)
        prepare_global_logging(serialization_dir, file_friendly_logging)

        serialization_params = deepcopy(params).as_dict(quiet=True)
        with open(os.path.join(serialization_dir, CONFIG_NAME), "w") as param_file:
            json.dump(serialization_params, param_file, indent=4)

        params.pop("model", None)
        vocabulary_params = params.pop("vocabulary", {})
        if vocabulary_params.get("directory_path", None):
            logger.warning(
                "You passed `directory_path` in parameters for the vocabulary in "
                "your configuration file, but it will be ignored. "
            )

        all_datasets = datasets_from_params(self, params)
        vocab = self.vocab

        if extend_vocab:
            datasets_for_vocab_creation = set(
                params.pop("datasets_for_vocab_creation", all_datasets)
            )

            for dataset in datasets_for_vocab_creation:
                if dataset not in all_datasets:
                    raise ConfigurationError(
                        f"invalid 'dataset_for_vocab_creation' {dataset}"
                    )

            logger.info(
                "Extending model vocabulary using %s data.",
                ", ".join(datasets_for_vocab_creation),
            )
            vocab.extend_from_instances(
                vocabulary_params,
                (
                    instance
                    for key, dataset in all_datasets.items()
                    for instance in dataset
                    if key in datasets_for_vocab_creation
                ),
            )

            self.extend_embedder_vocab(embedding_sources_mapping)

        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(self.vocab)

        train_data = all_datasets["train"]
        validation_data = all_datasets.get("validation")
        test_data = all_datasets.get("test")

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in self.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        # TODO: Customize trainer for better biome integration
        trainer = Trainer.from_params(
            model=self,
            serialization_dir=serialization_dir,
            iterator=iterator,
            train_data=train_data,
            validation_data=validation_data,
            params=trainer_params,
        )
        evaluate_on_test = params.pop_bool("evaluate_on_test", False)
        params.assert_empty("base train command")
        try:
            metrics = trainer.train()
        except KeyboardInterrupt:
            # if we have completed an epoch, try to create a model archive.
            if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
                logging.info(
                    "Fine-tuning interrupted by the user. Attempting to create "
                    "a model archive using the current best epoch weights."
                )
                archive_model(
                    serialization_dir, files_to_archive=params.files_to_archive
                )
            raise

        # Evaluate
        if test_data and evaluate_on_test:
            logger.info("The model will be evaluated using the best epoch weights.")
            test_metrics = evaluate(
                self,
                test_data,
                data_iterator=iterator,
                cuda_device=trainer._cuda_devices[
                    0
                ],  # pylint: disable=protected-access,
                batch_weight_key=batch_weight_key,
            )

            for key, value in test_metrics.items():
                metrics["test_" + key] = value

        elif test_data:
            logger.info(
                "To evaluate on the test set after training, pass the "
                "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
            )

        # Now tar up results
        archive_model(serialization_dir, files_to_archive=params.files_to_archive)

        metrics_json = json.dumps(metrics, indent=2)
        with open(os.path.join(serialization_dir, "metrics.json"), "w") as metrics_file:
            metrics_file.write(metrics_json)

        return self
