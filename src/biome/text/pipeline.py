import copy
import datetime
import glob
import inspect
import logging
import os
import time
import uuid
from inspect import Parameter
from threading import Thread
from typing import Any, Dict, List, Optional, Type, Union, cast
from urllib.error import URLError

import numpy
import uvicorn
import yaml
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Vocabulary
from allennlp.models import load_archive
from allennlp.models.archival import Archive
from biome.text._impl_model import AllennlpModel, _BaseModelImpl
from biome.text.configuration import (
    PipelineConfiguration,
    TrainerConfiguration,
    VocabularyConfiguration,
)
from biome.text.data import DataSource
from biome.text.errors import http_error_handling
from biome.text.helpers import (
    split_signature_params_by_predicate,
    update_method_signature,
)
from biome.text.ui import launch_ui
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient
from elasticsearch import Elasticsearch
from fastapi import FastAPI

from . import constants, helpers
from .model import Model
from .modules.encoders import TimeDistributedEncoder
from .modules.heads import TaskHead
from .modules.heads.defs import TaskHeadSpec

try:
    import ujson as json
except ModuleNotFoundError:
    import json

__default_impl__ = _BaseModelImpl


def __register(impl_class, overrides: bool = False):
    """Register the impl. class in allennlp components registry"""

    AllennlpModel.register(impl_class.__name__, exist_ok=overrides)(impl_class)
    DatasetReader.register(impl_class.__name__, exist_ok=overrides)(impl_class)


__register(__default_impl__, overrides=True)


class _ExploreConfiguration:
    """Configures an exploration run

    # Parameters
        batch_size: `int`
            The batch size for indexing predictions (default is `500)
        prediction_cache_size: `int`
            The size of the cache for caching predictions (default is `0)
        explain: `bool`
            Whether to extract and return explanations of token importance (default is `False`)
        force_delete: `bool`
            Whether to delete existing explore with `explore_id` before indexing new items (default is `True)
        metadata: `kwargs`
            Additional metadata to index in Elasticsearch
    """

    def __init__(
        self,
        batch_size: int = 500,
        prediction_cache_size: int = 0,
        explain: bool = False,
        force_delete: bool = True,
        **metadata,
    ):
        self.batch_size = batch_size
        self.prediction_cache = prediction_cache_size
        self.explain = explain
        self.force_delete = force_delete
        self.metadata = metadata


class _ElasticsearchExplore:
    """Elasticsearch data exploration class"""

    def __init__(self, es_index: str, es_host: Optional[str] = None):
        self.es_index = es_index
        self.es_host = es_host or constants.DEFAULT_ES_HOST
        if not self.es_host.startswith("http"):
            self.es_host = f"http://{self.es_host}"

        self.client = Elasticsearch(
            hosts=es_host, retry_on_timeout=True, http_compress=True
        )
        self.es_doc = helpers.get_compatible_doc_type(self.client)

    def create_explore_data_record(self, parameters: Dict[str, Any]):
        """Creates an exploration data record data exploration"""

        self.client.indices.create(
            index=constants.BIOME_METADATA_INDEX,
            body={
                "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}
            },
            params=dict(ignore=400),
        )

        self.client.update(
            index=constants.BIOME_METADATA_INDEX,
            doc_type=constants.BIOME_METADATA_INDEX_DOC,
            id=self.es_index,
            body={
                "doc": dict(
                    name=self.es_index, created_at=datetime.datetime.now(), **parameters
                ),
                "doc_as_upsert": True,
            },
        )

    def create_explore_data_index(self, force_delete: bool):
        """Creates an explore data index if not exists or is forced"""
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
            self.client.indices.delete(index=self.es_index, ignore=[400, 404])

        self.client.indices.create(
            index=self.es_index,
            body={"mappings": {self.es_doc: {"dynamic_templates": dynamic_templates}}},
            ignore=400,
        )


class _TrainConfiguration:
    """Configures a training run

    # Parameters
        output: `str`
             The experiment output path
        vocab: `vocab`
            The path to an existing vocabulary
        trainer_path: `str`
             The trainer file path
        train_cfg: `str`
            The train datasource file path
        validation_cfg: `Optional[str]`
            The validation datasource file path
        test_cfg: `Optional[str]`
            The test datasource file path
        verbose: `bool`
            Whether to show verbose logs (default is `False`)
        extend_vocab: `bool`
            Extends vocabulary namespaces with training data
    """

    def __init__(
        self,
        output: str,
        trainer: TrainerConfiguration,
        train_cfg: str = "",
        validation_cfg: Optional[str] = None,
        test_cfg: Optional[str] = None,
        verbose: bool = False,
    ):
        self.output = output
        self.trainer = trainer
        self.training = train_cfg
        self.validation = validation_cfg
        self.test = test_cfg
        self.verbose = verbose


class Pipeline:
    """Manages NLP models configuration and actions.

    Use `Pipeline` for creating new models from a configuration or loading a pre-trained model.

    Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.

    # Parameters
        pretrained_path: `Optional[str]`
            The path to the model.tar.gz of a pre-trained `Pipeline`
        config: `Optional[PipelineConfiguration]`
            A `PipelineConfiguration` object defining the configuration of the fresh `Pipeline`.
    """

    __LOGGER = logging.getLogger(__name__)
    __TRAINING_CACHE_DATA = "instances_data"

    # TODO: Signature makes you think you can pass both a pretrained_path and a config, while only one option possible.
    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        config: Optional[PipelineConfiguration] = None,
    ):

        self._binary = pretrained_path
        self._config = copy.deepcopy(config)

        if self._binary:
            archive = load_archive(self._binary)
            self._model = self.__model_from_archive(archive)
            self._config = self.__config_from_archive(archive)
        else:
            self._model = self.__model_from_config(self.config)

        if not isinstance(self._model, __default_impl__):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")

        self.__update_prediction_signatures()

    @classmethod
    def from_file(cls, path: str, vocab_config: VocabularyConfiguration) -> "Pipeline":
        """Creates a pipeline from a config yaml file path

        Parameters
        ----------
        path: `str`
            The path to a YAML configuration file
        vocab_config: `Optional[VocabularyConfiguration]`
            A `PipelineConfiguration` object defining the configuration of a fresh `Pipeline`.

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """
        with open(path) as yamL_file:
            return cls.from_config(yamL_file.read(), vocab_config=vocab_config)

    @classmethod
    def from_config(
        cls,
        config: Union[str, PipelineConfiguration],
        vocab_config: Optional[VocabularyConfiguration] = None,
    ) -> "Pipeline":
        """Creates a pipeline from a `PipelineConfiguration` object

        Parameters
        ----------
            config: `Union[str, PipelineConfiguration]`
                A `PipelineConfiguration` object or a YAML `str` for the pipeline configuration
            vocab_config: `Optional[VocabularyConfiguration]`
                A `VocabularyConfiguration` object for associating a vocabulary to the pipeline

        Returns
        -------
            pipeline: `Pipeline`
                A configured pipeline
        """

        if isinstance(config, str):
            config = PipelineConfiguration.from_params(Params(yaml.safe_load(config)))
        pipeline = cls(config=config)
        vocab = cls._load_vocabulary(pipeline, vocab_config)
        pipeline._model.update_vocab(vocab)

        return pipeline

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Pipeline":
        """Loads a pipeline from a pre-trained pipeline from a model.tar.gz file path

        # Parameters
            path: `str`
                The path to the model.tar.gz file of a pre-trained `Pipeline`

        # Returns
            pipeline: `Pipeline`
                A configured pipeline
        """
        return cls(pretrained_path=path)

    def train(
        self,
        output: str,
        trainer: TrainerConfiguration,
        training: str,
        validation: Optional[str] = None,
        test: Optional[str] = None,
        verbose: bool = False,
        extend_vocab: bool = False,
        restore: bool = True,
    ) -> "Pipeline":
        """Launches a training run with the specified configurations and datasources

        Parameters
        ----------
        output: `str`
            The experiment output path
        trainer: `str`
            The trainer file path
        training: `str`
            The train datasource file path
        validation: `Optional[str]`
            The validation datasource file path
        test: `Optional[str]`
            The test datasource file path
        verbose: `bool`
            Turn on verbose logs
        extend_vocab: `bool`
            Extends vocab tokens with training data
        restore: `bool`
            If enabled, tries to read previous training status from output folder and
            continues training process from it

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """

        def prepare_experiment_folder(output: str, restore: bool) -> None:
            """If output folder already exists, we automatically recover the generated vocab in this folder.

            Allows reuse the generated vocab if something went wrong in previous executions

            Parameters
            ----------
            output
                Path to the output folder
            restore: `bool`
                If False, drops all previous training states

            Returns
            -------
            is_recovered
                True if existing output folder is recovered, False if output folder does not exist.
            """
            if not os.path.isdir(output):
                return

            drop_patterns = [
                os.path.join(output, "*.json"),
                os.path.join(output, "**/events.out*"),
            ]

            if not restore:
                drop_patterns.append(os.path.join(output, "*.th"))
                drop_patterns.append(
                    os.path.join(output, self.__TRAINING_CACHE_DATA, "*")
                )

            for pattern in drop_patterns:
                for file in glob.glob(pattern, recursive=True):
                    os.remove(file)

        prepare_experiment_folder(output, restore)
        # TODO: we need make deserializable the feature dict (InmutableDict class)
        #  It fails otherwise
        # self._model.cache_data(os.path.join(output, self.__TRAINING_CACHE_DATA))
        if extend_vocab:
            vocab = self._extend_vocab_from_sources(
                self.model.vocab,
                sources=[ds for ds in [training, validation, test] if ds],
            )
            self._model.update_vocab(vocab)

        return _PipelineHelper.train(
            self,
            config=_TrainConfiguration(
                test_cfg=test,
                output=output,
                trainer=trainer,
                train_cfg=training,
                validation_cfg=validation,
                verbose=verbose,
            ),
        )

    def predict(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        """Predicts over some input data with current state of the model

        # Parameters
            args: `*args`
            kwargs: `**kwargs`

        # Returns
            predictions: `Dict[str, numpy.ndarray]`
                A dictionary containing the predictions and additional information
        """
        # TODO: Paco, what is the best way to document this, given that the signature is dynamic?
        self._model = self._model.eval()
        return self._model.predict(*args, **kwargs)

    def explain(self, *args, **kwargs) -> Dict[str, Any]:
        """Predicts over some input data with current state of the model and provides explanations of token importance.

        # Parameters
            args: `*args`
            kwargs: `**kwargs`

        # Returns
            predictions: `Dict[str, numpy.ndarray]`
                A dictionary containing the predictions with token importance calculated using IntegratedGradients
        """
        # TODO: Paco, what is the best way to document this, given that the signature is dynamic?
        return self._model.explain(*args, **kwargs)

    def save_vocab(self, path: str) -> None:
        """Save the pipeline vocabulary into a path"""
        self._model.vocab.save_to_files(path)

    def explore(
        self,
        ds_path: str,
        explore_id: Optional[str] = None,
        es_host: Optional[str] = None,
        batch_size: int = 500,
        prediction_cache_size: int = 0,
        # TODO: do we need caching for Explore runs as well or only on serving time?
        explain: bool = False,
        force_delete: bool = True,
        **metadata,
    ) -> dd.DataFrame:
        """Launches Explore UI for a given datasource with current model

        Running this method inside a an `IPython` notebook will try to render the UI directly in the notebook.

        Running this outside a notebook will try to launch the standalone web application.

        # Parameters
            ds_path: `str`
                The path to the configuration of a datasource
            explore_id: `Optional[str]`
                A name or id for this explore run, useful for running and keep track of several explorations
            es_host: `Optional[str]`
                The URL to the Elasticsearch host for indexing predictions (default is `localhost:9200`)
            batch_size: `int`
                The batch size for indexing predictions (default is `500)
            prediction_cache_size: `int`
                The size of the cache for caching predictions (default is `0)
            explain: `bool`
                Whether to extract and return explanations of token importance (default is `False`)
            force_delete: `bool`
                Deletes exploration with the same `explore_id` before indexing the new explore items (default is `True)

        # Returns
            pipeline: `Pipeline`
                A configured pipeline
        """
        config = _ExploreConfiguration(
            batch_size=batch_size,
            prediction_cache_size=prediction_cache_size,
            explain=explain,
            force_delete=force_delete,
            **metadata,
        )

        es_config = _ElasticsearchExplore(
            es_index=explore_id or str(uuid.uuid1()),
            es_host=es_host or constants.DEFAULT_ES_HOST,
        )

        explore_df = _PipelineHelper.explore(self, ds_path, config, es_config)
        self._show_explore(es_config)

        return explore_df

    def serve(self, port: int = 9998):
        """Launches a REST prediction service with current model in a specified port (default is `9998)

        # Parameters
            port: `int`
                The port to make available the prediction service
        """
        self._model = self._model.eval()
        return _PipelineHelper.serve(self, port)

    def set_head(self, type: Type[TaskHead], **params):
        """Sets a new task head for the pipeline

        Use this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.

        # Parameters
            type: `Type[TaskHead]`
                The `TaskHead` class to be set for the pipeline (e.g., `TextClassification`
            params: `**kwargs`
                The `TaskHead` specific parameters (e.g., classification head needs a `pooler` layer)
        """

        self._config.head = TaskHeadSpec(type=type.__name__, **params)
        self._model.head = self._config.head.compile(model=self.model)

    @property
    def name(self):
        """Gets pipeline  name"""
        return self._model.name

    @property
    def inputs(self) -> List[str]:
        """Gets pipeline input field names"""
        return self._model.inputs

    @property
    def output(self) -> str:
        """Gets pipeline output field names"""
        return self._model.output

    @property
    def model(self) -> Model:
        """Gets pipeline backbone model"""
        return self.head.model

    @property
    def head(self) -> TaskHead:
        """Gets pipeline task head"""
        return self._model.head

    @property
    def config(self) -> PipelineConfiguration:
        return self._config

    @property
    def trained_path(self) -> str:
        """Path to binary file when load from binary"""
        return self._binary

    @property
    def type_name(self) -> str:
        """The pipeline name. Equivalent to task head name"""
        return self.head.__class__.__name__

    @property
    def trainable_parameters(self) -> int:
        """
        Return the number of trainable parameters.

        This number could be change before an after a training process, since trainer could fix some of them.

        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def trainable_parameter_names(self) -> List[str]:
        """Returns the name of pipeline trainable parameters"""
        return [name for name, p in self._model.named_parameters() if p.requires_grad]

    @staticmethod
    def __model_from_config(
        config: PipelineConfiguration, **extra_params
    ) -> __default_impl__:
        """Creates a internal base model from pipeline configuration"""
        return __default_impl__.from_params(Params({"config": config}), **extra_params)

    def _show_explore(self, elasticsearch: _ElasticsearchExplore) -> None:
        """Shows explore ui for data prediction exploration"""

        def is_service_up(url: str) -> bool:
            import urllib.request

            try:
                status_code = urllib.request.urlopen(url).getcode()
                return 200 <= status_code < 400
            except URLError:
                return False

        def launch_ui_app() -> Thread:
            process = Thread(
                target=launch_ui,
                name="ui",
                kwargs=dict(es_host=elasticsearch.es_host, port=ui_port),
            )
            process.start()
            return process

        def show_notebook_explore(url: str):
            """Shows explore ui in a notebook cell"""
            from IPython.core.display import HTML, display

            iframe = f"<iframe src={url} width=100% height=840></iframe>"
            display(HTML(iframe))

        def show_browser_explore(url: str):
            """Shows explore ui in a web browser"""
            import webbrowser

            webbrowser.open(url)

        ui_port = 9999
        waiting_seconds = 1
        url = f"http://localhost:{ui_port}/projects/default/explore/{elasticsearch.es_index}"

        if not is_service_up(url):
            launch_ui_app()

        time.sleep(waiting_seconds)
        show_func = (
            show_notebook_explore
            if helpers.is_running_on_notebook()
            else show_browser_explore
        )
        show_func(url)

    def _extend_vocab_from_files(self, from_path: str) -> "Pipeline":
        """Load a vocab from a vocab folder path an set it to the currernt pipeline"""
        vocab = self._model.vocab
        self._model = self.__model_from_config(
            self.config, vocab=vocab.from_files(from_path)
        )
        return self

    @classmethod
    def _vocab_from_path(cls, from_path) -> Optional[Vocabulary]:
        try:
            return Vocabulary.from_files(from_path)
        except TypeError:
            return None
        except FileNotFoundError:
            cls.__LOGGER.warning("%s folder not found", from_path)
            return None

    def _extend_vocab_from_sources(
        self, vocab: Vocabulary, sources: List[str], **extra_args
    ) -> Vocabulary:
        """Extends an already created vocabulary from a list of source dictionary"""
        vocab.extend_from_instances(
            params=Params(extra_args),
            instances=[
                instance
                for data_source in sources
                for instance in self._model.read(data_source)
            ],
        )
        return vocab

    @staticmethod
    def __model_from_archive(archive: Archive) -> __default_impl__:
        if not isinstance(archive.model, __default_impl__):
            raise ValueError(f"Wrong pipeline model: {archive.model}")
        return cast(__default_impl__, archive.model)

    @staticmethod
    def __config_from_archive(archive: Archive) -> PipelineConfiguration:
        config = archive.config["model"]["config"]
        return PipelineConfiguration.from_params(config)

    def __update_prediction_signatures(self):
        """For interactive work-flows, fixes the predict signature to the model inputs"""
        new_signature = inspect.Signature(
            [
                Parameter(name=_input, kind=Parameter.POSITIONAL_OR_KEYWORD)
                for _input in self.inputs
            ]
        )

        for method in [self.predict, self.explain]:
            self.__setattr__(
                method.__name__, update_method_signature(new_signature, method)
            )

    @classmethod
    def _load_vocabulary(
        cls, pipeline: "Pipeline", vocab_config: Optional[VocabularyConfiguration]
    ) -> Optional[Vocabulary]:
        """
        Extends a data vocabulary from a given configuration
        Parameters
        ----------
        pipeline: ``Pipeline``
            The target pipeline
        vocab_config: ``VocabularyConfiguration``
            The vocab extension configuration

        Returns
        -------

        An extended ``Vocabulary`` using the provided configuration

        """

        if not vocab_config:
            return pipeline.model.vocab

        _vocab = cls._vocab_from_path(vocab_config.from_path) or pipeline.model.vocab

        if vocab_config.sources:
            _vocab = cls._extend_vocab_from_sources(
                pipeline,
                vocab=_vocab,
                sources=vocab_config.sources,
                max_vocab_size=vocab_config.max_vocab_size,
                min_count=vocab_config.min_count,
                pretrained_files=vocab_config.pretrained_files,
                only_include_pretrained_words=vocab_config.only_include_pretrained_words,
                min_pretrained_embeddings=vocab_config.min_pretrained_embeddings,
                tokens_to_add=vocab_config.tokens_to_add,
            )
        return _vocab


class _PipelineHelper:
    """Extra pipeline methods"""

    __LOGGER = logging.getLogger(__name__)

    @classmethod
    def serve(cls, pipeline: Pipeline, port: int):
        """Serves an pipeline as rest api"""

        def make_app() -> FastAPI:
            app = FastAPI()

            @app.post("/predict")
            async def predict(inputs: Dict[str, Any]):
                with http_error_handling():
                    return sanitize(pipeline.predict(**inputs))

            @app.post("/explain")
            async def explain(inputs: Dict[str, Any]):
                with http_error_handling():
                    return sanitize(pipeline.explain(**inputs))

            @app.get("/_config")
            async def config():
                with http_error_handling():
                    return pipeline.config.as_dict()

            @app.get("/_status")
            async def status():
                with http_error_handling():
                    return {"ok": True}

            return app

        uvicorn.run(make_app(), host="0.0.0.0", port=port)

    @classmethod
    def train(cls, pipeline: Pipeline, config: _TrainConfiguration):
        def allennlp_configuration(
            pipeline: Pipeline, config: _TrainConfiguration
        ) -> Dict[str, Any]:
            """Creates a allennlp configuration for pipeline train experiment configuration"""

            def trainer_configuration(trainer: TrainerConfiguration) -> Dict[str, Any]:
                """Creates trainer configuration dict"""
                __excluded_keys = [
                    "data_bucketing",
                    "batch_size",
                    "cache_instances",
                    "in_memory_batches",
                ]  # Data iteration attributes
                return {
                    k: v for k, v in vars(trainer).items() if k not in __excluded_keys
                }

            def iterator_configuration(
                pipeline: Pipeline, trainer: TrainerConfiguration
            ) -> Dict[str, Any]:
                """Creates a data iterator configuration"""

                def _forward_inputs() -> List[str]:
                    """
                    Calculate the required head.forward arguments. We use this method
                    for automatically generate data iterator sorting keys
                    """
                    required, _ = split_signature_params_by_predicate(
                        pipeline.head.forward,
                        lambda p: p.default == inspect.Parameter.empty,
                    )
                    return [p.name for p in required] or [None]

                iterator_config = {
                    "batch_size": trainer.batch_size,
                    "max_instances_in_memory": max(
                        trainer.batch_size * trainer.in_memory_batches,
                        trainer.batch_size,
                    ),
                    "cache_instances": trainer.cache_instances,
                    "type": "basic",
                }

                if trainer.data_bucketing:
                    iterator_config.update(
                        {
                            "sorting_keys": [
                                [
                                    _forward_inputs()[0],
                                    "list_num_tokens"
                                    if isinstance(
                                        pipeline.model.encoder, TimeDistributedEncoder
                                    )
                                    else "num_tokens",
                                ]
                            ],
                            "type": "bucket",
                        }
                    )

                return iterator_config

            base_config = {
                "config": pipeline.config.as_dict(),
                "type": __default_impl__.__name__,
            }
            allennlp_config = {
                "trainer": trainer_configuration(config.trainer),
                "iterator": iterator_configuration(pipeline, config.trainer),
                "dataset_reader": base_config,
                "model": base_config,
                "train_data_path": config.training,
                "validation_data_path": config.validation,
                "test_data_path": config.test,
            }
            return copy.deepcopy({k: v for k, v in allennlp_config.items() if v})

        pipeline._model.launch_experiment(
            params=Params(allennlp_configuration(pipeline, config)),
            serialization_dir=config.output,
        )  # pylint: disable=protected-access,

        return pipeline.__class__(
            pretrained_path=os.path.join(config.output, "model.tar.gz"),
            config=pipeline.config,
        )

    @classmethod
    def explore(
        cls,
        pipeline: Pipeline,
        ds_path: str,
        config: _ExploreConfiguration,
        elasticsearch: _ElasticsearchExplore,
    ) -> dd.DataFrame:
        if config.prediction_cache > 0:
            pipeline.init_predictions_cache(config.prediction_cache)

        data_source = DataSource.from_yaml(ds_path)
        ddf_mapped = data_source.to_mapped_dataframe()
        # this only makes really sense when we have a predict_batch_json method implemented ...
        n_partitions = max(1, round(len(ddf_mapped) / config.batch_size))

        # a persist is necessary here, otherwise it fails for n_partitions == 1
        # the reason is that with only 1 partition we pass on a generator to predict_batch_json
        ddf_mapped = ddf_mapped.repartition(npartitions=n_partitions).persist()

        apply_func = pipeline.explain if config.explain else pipeline.predict

        ddf_mapped["annotation"] = ddf_mapped[pipeline.inputs].apply(
            lambda x: sanitize(apply_func(**x.to_dict())), axis=1, meta=(None, object)
        )

        ddf_source = (
            data_source.to_dataframe().repartition(npartitions=n_partitions).persist()
        )
        ddf_mapped["metadata"] = ddf_source.map_partitions(
            lambda df: df.to_dict(orient="records")
        )

        # TODO @dcfidalgo we could calculate base metrics here (F1, recall & precision) using dataframe.
        #  And include as part of explore metadata
        #  Does it's simple???

        ddf = DaskElasticClient(
            host=elasticsearch.es_host, retry_on_timeout=True, http_compress=True
        ).save(ddf_mapped, index=elasticsearch.es_index, doc_type=elasticsearch.es_doc)

        elasticsearch.create_explore_data_index(force_delete=config.force_delete)
        elasticsearch.create_explore_data_record(
            {
                **(config.metadata or {}),
                "datasource": ds_path,
                # TODO this should change when ui is normalized (action detail and action link naming)F
                "explore_name": elasticsearch.es_index,
                "model": pipeline.name,
                "columns": ddf.columns.values.tolist(),
                "metadata_columns": data_source.to_dataframe().columns.values.tolist(),
                "pipeline": pipeline.type_name,
                "output": pipeline.output,
                "inputs": pipeline.inputs,  # backward compatibility
                "signature": pipeline.inputs + [pipeline.output],
                "predict_signature": pipeline.inputs,
                "labels": pipeline.head.labels,
                "task": pipeline.head.task_name().as_string(),
            }
        )
        return ddf.persist()
