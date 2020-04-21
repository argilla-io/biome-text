import inspect
import json
import logging
import os
import pickle
import warnings
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

import numpy
import torch
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.models import Model as AllennlpModel
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
        **extras
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
                    or vocabulary.empty_vocab(features=config.features.compile()),
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
