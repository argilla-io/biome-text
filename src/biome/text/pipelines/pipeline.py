import copy
import datetime
import logging
import os
import pickle
import re
import warnings
from copy import deepcopy
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tempfile import mktemp
from typing import cast, Type, Optional, List, Dict, Tuple, Any, Generic, TypeVar, Union
from gevent.pywsgi import WSGIServer

import allennlp
import numpy
import pandas as pd
import yaml
from allennlp.common import JsonDict, Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.interpret import SaliencyInterpreter
from allennlp.models import Archive, Model
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.predictors import Predictor
from allennlp.service import server_simple

from dask import dataframe as dd
from dask_elk.client import DaskElasticClient
from flask_cors import CORS
from overrides import overrides
from torch.nn import LSTM

from biome.data import DataSource
from biome.text import constants
from biome.text.defs import TextClassifierPipeline, Pipeline
from biome.text.pipelines._impl.allennlp.dataset_readers import DataSourceReader
from biome.text.pipelines._impl.allennlp.learn import learn
from biome.text.pipelines._impl.allennlp.models import (
    load_archive,
    SequenceClassifierBase,
)
from biome.text.pipelines._impl.allennlp.predictors.utils import (
    get_predictor_from_archive,
)
from biome.text.pipelines.defs import ExploreConfig, ElasticsearchConfig

try:
    import ujson as json
except ModuleNotFoundError:
    import json


class _HashDict(dict):
    """
    hashable dict implementation.
    BE AWARE! Since dicts are mutable, the hash can change!
    """

    def __hash__(self):
        # user a better way
        return pickle.dumps(self).__hash__()


Architecture = TypeVar("Architecture", bound=allennlp.models.Model)
Reader = TypeVar("Reader", bound=DataSourceReader)


class Pipeline(Generic[Architecture, Reader], Predictor, TextClassifierPipeline):
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

    _LOGGER = logging.getLogger(__name__)

    # Disable logging
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("allennlp").setLevel(logging.WARNING)

    PIPELINE_FIELD = "pipeline"
    ARCHITECTURE_FIELD = "architecture"
    TYPE_FIELD = "type"

    PREDICTION_FILE_NAME = "predictions.json"

    def __init__(self, model: Architecture, reader: Reader):
        super(Pipeline, self).__init__(model, reader)
        self.__config = {}
        self.__binary_path = None
        self.__prediction_logger = None

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

        self.__prediction_logger = predictions_logger

    @classmethod
    def by_name(cls: Type["Pipeline"], name: str) -> Type["Pipeline"]:
        return cast(Type[Pipeline], Predictor.by_name(name))

    @classmethod
    def reader_class(cls) -> Type[Reader]:
        """
        Must be implemented by subclasses

        Returns
        -------
            The class of ``DataSourceReader`` used in the model instance
        """
        _, reader = cls.__resolve_generics()
        return reader

    @classmethod
    def __resolve_generics(cls) -> Tuple[Type[Architecture], Type[Reader]]:
        return getattr(cls, "__orig_bases__")[0].__args__

    @classmethod
    def model_class(cls) -> Type[Architecture]:
        """
        Must be implemented by subclasses

        Returns
        -------
            The class of ``allennlp.models.Model`` used in the model instance
        """
        model, _ = cls.__resolve_generics()
        return model

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
        model_name = (
            os.path.basename(os.path.dirname(self.__binary_path))
            if self.__binary_path
            else "empty"
        )
        return f"{self.__module__}.{self.__class__.__name__}::{model_name}"

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

    def predict(self, *args, **inputs) -> dict:
        return self.predict_json(inputs)

    def _update_binary_path(self, path) -> None:
        if not self.__binary_path:
            self.__binary_path = path

    def _update_config(self, config) -> None:
        self.__config = config

    @classmethod
    def load(cls, binary_path: str, **kwargs) -> "Pipeline":
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
            allennlp.models.Model.register(name, exist_ok=True)(cls.model_class())
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
        hashable_dict = _HashDict(inputs)

        output = self._predict_hashable_json(hashable_dict)

        if self.__prediction_logger:
            self.__prediction_logger.info(
                json.dumps(dict(inputs=inputs, annotation=output))
            )

        return output

    def _predict_hashable_json(self, inputs: _HashDict) -> Optional[JsonDict]:
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
            warnings.warn(
                "Prediction cache already initiated!", category=RuntimeWarning
            )
            return

        decorated_func = lru_cache(maxsize=max_size)(self._predict_hashable_json)

        self.__setattr__("_predict_hashable_json", decorated_func)

    def inputs_keys(self) -> List[str]:
        return [k for k, v in self.signature.items() if not v.get("optional")]

    def output(self) -> str:
        return "label"

    def explore(
        self, ds_path: str, config: ExploreConfig, es_config: ElasticsearchConfig
    ) -> dd.DataFrame:
        from biome.text.pipelines._impl.allennlp.interpreters import (
            IntegratedGradient as DefaultInterpreterClass,
        )

        def _interpret_dataframe(
            df: pd.DataFrame,
            pipeline: Pipeline,
            interpreter_klass: Type = DefaultInterpreterClass,
        ) -> pd.Series:
            """
            Apply a model interpretation to every partition dataframe

            Parameters
            ----------
            df: pd.DataFrame
                The partition DataFrame
            pipeline: str
                The pipeline
            interpreter_klass: Type
                The used interpreted class

            Returns
            -------

            A pandas Series representing the interpretations

            """

            def interpret_row(
                row: pd.Series, interpreter: SaliencyInterpreter
            ) -> Union[dict, List[dict]]:
                """Interpret a incoming dataframe row"""
                data = row.to_dict()
                interpretation = interpreter.saliency_interpret_from_json(data)
                if len(interpretation) == 0:
                    return {}
                if len(interpretation) == 1:
                    return interpretation[0]
                return interpretation

            interpreter = interpreter_klass(pipeline)
            return df.apply(interpret_row, interpreter=interpreter, axis=1)

        if config.prediction_cache > 0:
            self.init_prediction_cache(config.prediction_cache)
        data_source = DataSource.from_yaml(ds_path)
        ddf_mapped = data_source.to_mapped_dataframe()
        # this only makes really sense when we have a predict_batch_json method implemented ...
        n_partitions = max(1, round(len(ddf_mapped) / config.batch_size))

        # a persist is necessary here, otherwise it fails for n_partitions == 1
        # the reason is that with only 1 partition we pass on a generator to predict_batch_json
        ddf_mapped = ddf_mapped.repartition(npartitions=n_partitions).persist()
        ddf_mapped_columns = ddf_mapped.columns

        ddf_mapped["annotation"] = ddf_mapped[self.inputs_keys()].apply(
            lambda x: sanitize(self.predict(**x.to_dict())), axis=1, meta=(None, object)
        )

        if config.interpret:
            # TODO we should apply the same mechanism for the model predictions. Creating a new pipeline
            #  for every partition
            ddf_mapped["interpretations"] = ddf_mapped[
                ddf_mapped_columns
            ].map_partitions(_interpret_dataframe, pipeline=self, meta=(None, object))

        ddf_source = data_source.to_dataframe()
        ddf_source = ddf_source.repartition(npartitions=n_partitions).persist()

        # We are sure that both data frames are aligned!
        # A 100% safe way would be to set_index of both data frames on a meaningful column.
        # The main problem are multiple csv files (read_csv("*.csv")), where the index starts from 0 for each file ...
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ddf = dd.concat([ddf_source, ddf_mapped], axis=1)

        # TODO @dcfidalgo we could calculate base metrics here (F1, recall & precision) using dataframe.
        #  And include as part of explore metadata
        #  Does it's simple???

        ddf = DaskElasticClient(
            host=es_config.es_host, retry_on_timeout=True, http_compress=True
        ).save(ddf, index=es_config.es_index, doc_type=es_config.es_doc)

        merged_metadata = {
            **(config.metadata or {}),
            "datasource": ds_path,
            # TODO this should change when ui is normalized (action detail and action link naming)F
            "explore_name": es_config.es_index,
            "model": self.name,
            "columns": ddf.columns.values.tolist(),
        }

        self.register_biome_prediction(
            name=es_config.es_index, es_config=es_config, **merged_metadata
        )
        self.__prepare_es_index(es_config, force_delete=config.force_delete)
        ddf = ddf.persist()
        self._LOGGER.info(
            "Data annotated successfully. You can explore your data here: %s",
            f"{constants.EXPLORE_APP_ENDPOINT}/projects/default/explore/{es_config.es_index}",
        )

        return ddf

    def serve(self, port: int, predictions: str):
        if predictions:
            self.init_prediction_logger(predictions)

        app = server_simple.make_app(self, title=self.name)
        CORS(app)

        http_server = WSGIServer(("0.0.0.0", port), app)
        self._LOGGER.info("Model loaded, serving on port %s", port)

        http_server.serve_forever()

    def register_biome_prediction(
        self, name: str, es_config: ElasticsearchConfig, **kwargs
    ):
        """
        Creates a new metadata entry for the incoming prediction

        Parameters
        ----------
        name
            A descriptive prediction name
        pipeline
            The pipeline used for the prediction batch
        es_config:
            The Elasticsearch configuration data
        kwargs
            Extra arguments passed as extra metadata info
        """

        metadata_index = constants.BIOME_METADATA_INDEX

        es_config.client.indices.create(
            index=metadata_index,
            body={
                "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}
            },
            params=dict(ignore=400),
        )

        parameters = {
            **kwargs,
            "pipeline": self.name,
            "signature": self.inputs_keys() + [self.output()],
            "predict_signature": self.inputs_keys(),
            # TODO remove when ui is adapted
            "inputs": self.inputs_keys(),  # backward compatibility
        }

        es_config.client.update(
            index=metadata_index,
            doc_type=es_config.es_doc,
            id=es_config.es_index,
            body={
                "doc": dict(
                    name=name, created_at=datetime.datetime.now(), **parameters
                ),
                "doc_as_upsert": True,
            },
        )

    @staticmethod
    def __prepare_es_index(es_config: ElasticsearchConfig, force_delete: bool):
        dynamic_templates = [
            {
                data_type: {
                    "match_mapping_type": data_type,
                    "path_match": path_match,
                    "mapping": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                }
            }
            for data_type, path_match in [("*", "*.value"), ("string", "*")]
        ]

        if force_delete:
            es_config.client.indices.delete(index=es_config.es_index, ignore=[400, 404])

        es_config.client.indices.create(
            index=es_config.es_index,
            body={
                "mappings": {es_config.es_doc: {"dynamic_templates": dynamic_templates}}
            },
            ignore=400,
        )

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

    @staticmethod
    def _empty_vocab(labels: List[str] = None) -> Vocabulary:
        """
        This method generate a mock vocabulary for the 3 common allennlp namespaces.
        If default model use another tokens indexer key name, the pipeline model won't be loaded
        from configuration
        """
        labels = labels or ["true", "false"]
        vocab = Vocabulary()

        vocab.add_tokens_to_namespace(labels, namespace="labels")
        for namespace in ["tokens", "tokens_characters"]:
            vocab.add_token_to_namespace("t", namespace=namespace)

        return vocab

    @classmethod
    def empty_pipeline(cls, labels: List[str]) -> "Pipeline":
        """Creates a dummy pipeline with labels for model layers"""
        vocab = cls._empty_vocab(labels)
        return cls(
            model=cls.model_class()(
                text_field_embedder=BasicTextFieldEmbedder(
                    token_embedders={
                        "tokens": Embedding.from_params(
                            vocab=vocab,
                            params=Params({"embedding_dim": 64, "trainable": True}),
                        )
                    }
                ),
                seq2vec_encoder=PytorchSeq2VecWrapper(
                    LSTM(
                        input_size=64,
                        hidden_size=32,
                        bidirectional=True,
                        batch_first=True,
                    )
                ),
                vocab=vocab,
            ),
            reader=cls.reader_class()(
                token_indexers={"tokens": SingleIdTokenIndexer()}
            ),
        )

    @classmethod
    def from_config(cls, path: str, labels: List[str] = None) -> "Pipeline":
        """
        Read a ``Pipeline`` subclass instance by reading a configuration file

        Parameters
        ----------
        path
            The configuration file path
        labels:
            Optional. If passed, set a list of output labels for empty pipeline model

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
        name = cls.__get_pipeline_name_from_config(data)

        try:
            model = Model.from_params(
                params=Params(Pipeline.__get_model_params(data, name)),
                vocab=cls._empty_vocab(labels),
            )
        except allennlp.common.checks.ConfigurationError:
            model = None

        # Creating an empty pipeline
        model = pipeline_class(
            model=model,
            reader=cast(
                DataSourceReader,
                DatasetReader.from_params(
                    Params(Pipeline.__get_reader_params(data, name))
                ),
            ),
        )
        # Include pipeline configuration
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

    def __del__(self):
        if hasattr(self._predict_hashable_json, "cache_info"):
            # pylint: disable=no-member
            self._LOGGER.info(
                "Cache statistics: %s", self._predict_hashable_json.cache_info()
            )

    def extend_labels(self, labels: List[str]) -> None:
        """Allow extend prediction labels to pipeline"""
        if not isinstance(self.model, SequenceClassifierBase):
            warnings.warn(f"Model {self.model} is not updatable")
        else:
            cast(SequenceClassifierBase, self.model).extend_labels(labels)

    def get_output_labels(self) -> List[str]:
        """Output model labels"""
        if not isinstance(self.model, SequenceClassifierBase):
            warnings.warn(f"get_output_labels not suported for model {self.model}")
            return []
        return cast(SequenceClassifierBase, self.model).output_classes
