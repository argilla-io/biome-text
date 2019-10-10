import logging
import os
import re
from tempfile import mktemp
from typing import cast, Type

import allennlp
import yaml
from allennlp.common import JsonDict, Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
from allennlp.models import Archive
from allennlp.predictors import Predictor
from overrides import overrides
import biome
from biome.text.dataset_readers.datasource_reader import DataSourceReader
from biome.text.models import load_archive

class BaseModelInstance(Predictor):
    """
    This class combine the different allennlp components that make possible a "BaseModelInstance",
    understanding as a model, not only the definition of the neural network architecture,
    but also the transformation of the input data to Instances and the evaluation of
    predictions on new data

    The base idea is that this class contains the model and the dataset reader (as a predictor does),
    and allow operations of learning, predict and save

    Parameters
    ----------
    model
        The class:~allennlp.models.BaseModelInstance architecture

    reader
        The class:allennlp.data.DatasetReader
    """

    _logger = logging.getLogger(__name__)

    # Disable allennlp logging
    logging.getLogger("allennlp").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)

    def __init__(self, model: allennlp.models.model.Model, reader: DataSourceReader):
        super(BaseModelInstance, self).__init__(model, reader)
        self.__config = {}

    @property
    def pipeline(self) -> "biome.text.dataset_readers.DataSourceReader":
        """
        The data pipeline (AKA ``DatasetReader``)

        Returns
        -------
            The configured ``DatasetReader``

        """
        return self._dataset_reader

    @property
    def architecture(self) -> allennlp.models.Model:
        """
        The model architecture (AKA ``allennlp.models.BaseModelInstance``)

        Returns
        -------
            The configured ``allennlp.models.BaseModelInstance``
        """
        return self._model

    @property
    def allennlp_config(self) -> dict:
        """
        A representation of reader and model in a properties defined way
        as allennlp does

        Returns
        -------
            The configuration dictionary
        """
        return self.__config.copy()

    def predict(self, **inputs) -> dict:
        return self.predict_json(inputs)

    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseModelInstance":
        name = cls.__to_snake_case(cls.__name__)
        Predictor.register(name, exist_ok=True)(cls)
        archive = load_archive(path, **kwargs)
        return cls.from_archive(archive, predictor_name=name)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self.pipeline.text_to_instance_with_data_filter(json_dict)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        from allennlp.common.util import sanitize

        instance = self._json_to_instance(inputs)
        output = self.architecture.forward_on_instance(instance)
        return sanitize(output)

    @staticmethod
    def __to_snake_case(name):
        """
        A helper method for convert a CamelCase name into a snake_case name

        Parameters
        ----------
        name
            The original name

        Returns
        -------
            The corresponding snake_case name

        """
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @staticmethod
    def yaml_to_dict(filepath: str):
        with open(filepath) as yaml_content:
            config = yaml.safe_load(yaml_content)
        return config

    @classmethod
    def from_config(cls, path: str) -> "BaseModelInstance":
        """
        Read a ``BaseModelInstance`` subclass instance by reading a configuration file

        Parameters
        ----------
        path
            The configuration file path

        Returns
        -------
            An instance of ``BaseModelInstance`` with no architecture, since the internal
            ``allennlp.models.Model`` needs a Vocabulary for the initialization

        """
        data = cls.yaml_to_dict(path)
        # backward compatibility
        if data.get("topology"):
            data = data["topology"]
        main_class = cast(
            Type[BaseModelInstance], Predictor.by_name(data.get("type", cls.__name__))
        )
        name = cls.__to_snake_case(main_class.__name__)

        model = main_class(
            model=None,
            reader=cast(
                DataSourceReader,
                DatasetReader.from_params(
                    Params({**data["pipeline"], "type": name}.copy())
                ),
            ),
        )
        model.__config = {
            "dataset_reader": {**data["pipeline"], "type": name},
            "model": {**data["architecture"], "type": name},
        }

        return model

    def learn(self, trainer: str, train: str, validation: str, output: str):
        """
        Launch a learning process for loaded model configuration.

        Once the learn process finish, the model is ready for make predictions

        Parameters
        ----------

        trainer
            The trainer file path
        train
            The train datasource file path

        validation
            The validation datasource file path

        output
            The learn output path

        """
        from biome.text.commands.learn import learn

        spec = mktemp()
        with open(spec, "wt") as file:
            yaml.safe_dump(self.allennlp_config, file)

        _ = learn.learn(
            output=output,
            model_spec=spec,
            trainer_path=trainer,
            train_cfg=train,
            validation_cfg=validation,
        )

        model = self.load(os.path.join(output, "model.tar.gz"))

        self._model = model.architecture
        self._dataset_reader = model.pipeline

    @classmethod
    def _load_callback(cls, archive: Archive, reader: DatasetReader):

        """
        This method allow manage custom loads when the general way doesn't work

        Parameters
        ----------
        archive
            The loaded archive
        reader
            The corresponding DatasetReader

        Returns
        -------

        """
        raise ConfigurationError(
            "Cannot load sequence classifier without pipeline configuration"
        )
