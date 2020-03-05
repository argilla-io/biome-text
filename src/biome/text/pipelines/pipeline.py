import copy
import logging
import os
import re
import warnings
from copy import deepcopy
from functools import lru_cache
from tempfile import mktemp
from typing import cast, Type, Optional, List, Dict, Tuple, Any

import allennlp
import numpy
import yaml
from allennlp.common import JsonDict, Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import LabelField
from allennlp.models import Archive, Model
from allennlp.predictors import Predictor
from biome.text.dataset_readers.datasource_reader import DataSourceReader
from biome.text.models import load_archive
from biome.text.utils import HashDict
from biome.text.pipelines.learn.allennlp import learn
from biome.text.predictors.utils import get_predictor_from_archive
from overrides import overrides


class Pipeline(Predictor):
    """
    This class combine the different allennlp components that make possible a ``Pipeline`,
    understanding as a model, not only the definition of the neural network architecture,
    but also the transformation of the input data to Instances and the evaluation of
    predictions on new data

    The base idea is that this class contains the model and the dataset reader (as a predictor does),
    and allow operations of learning, predict and save

    Parameters
    ----------
    model`
        The class:~allennlp.models.Model architecture

    reader
        The class:allennlp.data.DatasetReader
    """

    _logger = logging.getLogger(__name__)

    # Disable allennlp logging
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)

    PIPELINE_FIELD = "pipeline"
    ARCHITECTURE_FIELD = "architecture"
    TYPE_FIELD = "type"

    def __init__(self, model: allennlp.models.model.Model, reader: DataSourceReader):
        super(Pipeline, self).__init__(model, reader)
        self.__config = {}
        self.__binary_path = None

    @classmethod
    def reader_class(cls) -> Type[DataSourceReader]:
        """
        Must be implemented by subclasses

        Returns
        -------
            The class of ``DataSourceReader`` used in the model instance
        """
        raise NotImplementedError

    @classmethod
    def model_class(cls) -> Type[allennlp.models.Model]:
        """
        Must be implemented by subclasses

        Returns
        -------
            The class of ``allennlp.models.Model`` used in the model instance
        """
        raise NotImplementedError

    @property
    def reader(self) -> DataSourceReader:
        """
        The data reader (AKA ``DatasetReader``)

        Returns
        -------
            The configured ``DatasetReader``

        """
        return self._dataset_reader

    @property
    def model(self) -> allennlp.models.Model:
        """
        The model (AKA ``allennlp.models.Model``)

        Returns
        -------
            The configured ``allennlp.models.Model``
        """
        return self._model

    @property
    def name(self) -> str:
        """
        Get the pipeline name

        Returns
        -------
            The fully qualified pipeline class name
        """
        return f"{self.__module__}.{self.__class__.__name__}"

    @property
    def config(self) -> dict:
        """
        A representation of reader and model in a properties defined way
        as allennlp does

        Returns
        -------
            The configuration dictionary
        """
        return self.__config.copy()

    @property
    def signature(self) -> dict:
        """
        Describe de input signature for the pipeline

        Returns
        -------
            A dict of expected inputs
        """
        return self._dataset_reader.signature

    def predict(self, **inputs) -> dict:
        return self.predict_json(inputs)

    def _update_binary_path(self, path) -> None:
        if not self.__binary_path:
            self.__binary_path = path

    def _update_config(self, config) -> None:
        self.__config = config

    @classmethod
    def load(
        cls, binary_path: str, **kwargs
    ) -> "Pipeline":
        """Load a model pipeline form a binary path.

        Parameters
        ----------
        binary_path
            Path to the binary file
        kwargs
            Passed on to the biome.text.models.load_archive method

        Returns
        -------
        pipeline
        """
        # TODO: Read labels from tar.gzs
        name = None
        # TODO resolve load from Pipeline.class. By now, you must decorate your own
        #  pipeline classes as an :class:~`allennlp.predictors.Predictor`
        if cls != Pipeline:
            name = cls.__registrable_name(cls)
            # TODO From now, we cannot pass the fully qualified class name as type parameter. We have an open
            #  PR for that. See (https://github.com/allenai/allennlp/pull/3344)
            #  So, we register the required components by allennlp before load them
            Predictor.register(name, exist_ok=True)(cls)
            Model.register(name, exist_ok=True)(cls.model_class())
            DatasetReader.register(name, exist_ok=True)(cls.reader_class())

        archive = load_archive(binary_path, **kwargs)
        predictor = get_predictor_from_archive(archive, predictor_name=name)
        pipeline = cast(Pipeline, predictor)
        pipeline._update_binary_path(binary_path)

        return pipeline

    @classmethod
    def __registrable_name(cls, _class: Type["Pipeline"]) -> str:
        return cls.__to_snake_case(_class.__name__)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self.reader.text_to_instance(**json_dict)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:

        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs["logits"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))

        return [new_instance]

    @overrides
    def get_gradients(
        self, instances: List[Instance]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        Parameters
        ----------
        instances: List[Instance]

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
        The first item is a Dict of gradient entries for each input.
        The keys have the form  ``{grad_input_1: ..., grad_input_2: ... }``
        up to the number of inputs given. The second item is the model's output.

        Notes
        -----
        Takes a ``JsonDict`` representing the inputs of the model and converts
        them to :class:`~allennlp.data.instance.Instance`s, sends these through
        the model :func:`forward` function after registering hooks on the embedding
        layer of the model. Calls :func:`backward` on the loss and then removes the
        hooks.
        """
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        outputs = self._model.decode(self._model.forward(**dataset.as_tensor_dict()))

        loss = outputs["loss"]
        self._model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        embedding_gradients.reverse()
        grads = [grad.detach().cpu().numpy() for grad in embedding_gradients]
        return grads, outputs

    @overrides
    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        """
        Converts incoming json to a :class:`~allennlp.data.instance.Instance`,
        runs the model on the newly created instance, and adds labels to the
        :class:`~allennlp.data.instance.Instance`s given by the model's output.
        Returns
        -------
        List[instance]
        A list of :class:`~allennlp.data.instance.Instance`
        """
        # pylint: disable=assignment-from-no-return
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    @overrides
    def predict_json(self, inputs: JsonDict) -> Optional[JsonDict]:
        """Predict an input with the pipeline's model.

        Parameters
        ----------
        inputs
            The input features/tokens in form of a json dict

        Returns
        -------
        output
            The model's prediction in form of a dict.
            Returns None if the input could not be transformed to an instance.
        """
        hashable_dict = HashDict(inputs)

        return self._predict_hashable_json(hashable_dict)

    def _predict_hashable_json(self, inputs: HashDict) -> Optional[JsonDict]:
        """Predict an input with the pipeline's model with a hashable input to be able to cache the return value.

        Parameters
        ----------
        inputs
            The input features/tokens in form of a hashable dict

        Returns
        -------
        output
            The model's prediction in form of a dict.
            Returns None if the input could not be transformed to an instance.
        """
        instance = self._json_to_instance(inputs)
        if instance is None:
            return None

        output = sanitize(self.model.forward_on_instance(instance))
        return output

    def init_prediction_cache(self, max_size) -> None:
        """Initialize a prediction cache using the functools.lru_cache decorator

        Parameters
        ----------
        max_size
            Save up to max_size most recent items.
        """
        if hasattr(self._predict_hashable_json, "cache_info"):
            warnings.warn("Prediction cache already initiated!", category=RuntimeWarning)
            return

        decorated_func = lru_cache(maxsize=max_size)(self._predict_hashable_json)

        self.__setattr__("_predict_hashable_json", decorated_func)

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
        snake_case_pattern = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake_case_pattern).lower()

    @staticmethod
    def yaml_to_dict(filepath: str):
        with open(filepath) as yaml_content:
            config = yaml.safe_load(yaml_content)
        return config

    @classmethod
    def from_config(cls, path: str) -> "Pipeline":
        """
        Read a ``Pipeline`` subclass instance by reading a configuration file

        Parameters
        ----------
        path
            The configuration file path

        Returns
        -------
            An instance of ``Pipeline`` with no architecture, since the internal
            ``allennlp.models.Model`` needs a Vocabulary for the initialization

        """
        data = cls.yaml_to_dict(path)
        # backward compatibility
        if data.get("topology"):
            data = data["topology"]

        pipeline_class = cls.__get_pipeline_class(data)
        name = cls.__get_pipeline_name_from_config(data) or cls.__registrable_name(
            pipeline_class
        )
        # Creating an empty pipeline
        model = pipeline_class(
            model=None,
            reader=cast(
                DataSourceReader,
                DatasetReader.from_params(
                    Params(Pipeline.__get_reader_params(data, name))
                ),
            ),
        )
        # Include pipeline configuration
        # TODO This configuration will fail if the reader and model are registered with other names than the calculated
        #  registrable_name
        config = cls.yaml_to_dict(path)
        config[cls.PIPELINE_FIELD] = Pipeline.__get_reader_params(data, name)
        config[cls.ARCHITECTURE_FIELD] = Pipeline.__get_model_params(data, name)
        model._update_config(config)
        return model

    @classmethod
    def __get_reader_params(cls, data: dict, name: Optional[str] = None) -> dict:
        # TODO dataset_reader will not be supported as part of configuration definition
        config = data.get(cls.PIPELINE_FIELD, data.get("dataset_reader"))
        if name and not config.get(cls.TYPE_FIELD):
            config[cls.TYPE_FIELD] = name
        return copy.deepcopy(config)

    @classmethod
    def __get_model_params(cls, data: dict, name: Optional[str] = None) -> dict:
        # TODO model will not be supported as part of configuration definition
        config = data.get(cls.ARCHITECTURE_FIELD, data.get("model"))
        if name:
            config[cls.TYPE_FIELD] = name
        return copy.deepcopy(config)

    @classmethod
    def __get_pipeline_class(cls, config: dict) -> Type["Pipeline"]:
        """
        If we don't known the target class to load, we need keep class info in data configuration.

        Parameters
        ----------
        config

        Returns
        -------
            The real ``Pipeline`` subclass to be instantiated
        """
        if cls != Pipeline:
            return cls

        pipeline_type = cls.__get_pipeline_name_from_config(config)
        the_class = Predictor.by_name(pipeline_type)
        return cast(Type[Pipeline], the_class)

    @classmethod
    def __get_pipeline_name_from_config(cls, config: Dict[str, Any]):
        pipeline_type = config.get(
            cls.TYPE_FIELD, cls.__get_model_params(config).get(cls.TYPE_FIELD)
        )
        if not pipeline_type:
            raise ConfigurationError(
                "Cannot load the pipeline: No pipeline type found in file."
                "\nPlease, include the class property in your file or try to load configuration "
                "with your class directly: MyPipeline.from_config(config_file)"
            )
        return pipeline_type

    def learn(
        self,
        trainer: str,
        train: str,
        output: str,
        validation: str = None,
        test: Optional[str] = None,
        vocab: Optional[str] = None,
        verbose: bool = False,
    ):
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
        vocab: Vocab
            The already generated vocabulary path
        test: str
            The test datasource configuration
        verbose
            Turn on verbose logs
        """

        kwargs = dict(
            vocab=vocab,
            test_cfg=test,
            output=output,
            trainer_path=trainer,
            train_cfg=train,
            validation_cfg=validation,
            verbose=verbose,
        )

        if self.__binary_path:
            learn(model_binary=self.__binary_path, **kwargs)
        else:
            spec = mktemp()
            with open(spec, "wt") as file:
                yaml.safe_dump(self.config, file)
            _ = learn(model_spec=spec, **kwargs)

        model = self.load(os.path.join(output, "model.tar.gz"))
        self._model = model.model
        self._dataset_reader = model.reader

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
