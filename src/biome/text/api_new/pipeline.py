import copy
import inspect
import logging
import os
import shutil
import time
import uuid
from inspect import Parameter
from threading import Thread
from typing import Any, Dict, List, Optional, Type, Union, cast
from urllib.error import URLError

import numpy
import uvicorn
import yaml
from allennlp.commands.fine_tune import fine_tune_model
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader
from allennlp.models import load_archive
from allennlp.models.archival import Archive
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient
from fastapi import FastAPI

from biome.text.api_new.modules.encoders import TimeDistributedEncoder
from biome.text.api_new.modules.heads import TaskHead
from . import constants, helpers
from ._impl_model import AllennlpModel, _BaseModelImpl
from .configuration import PipelineConfiguration
from .data import DataSource
from .errors import http_error_handling
from .helpers import (
    ElasticsearchExplore,
    get_env_cuda_device,
    split_signature_params_by_predicate,
    update_method_signature,
    yaml_to_dict,
)
from .model import Model
from .modules.heads.defs import TaskHeadSpec
from ..commands.ui.ui import launch_ui

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


class VocabularyConfiguration:
    """Vocabulary creation configuration on pipeline loading from file configuration"""

    def __init__(
        self,
        sources: List[str],
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
    ):
        self.sources = sources
        self.pretrained_files = pretrained_files
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.only_include_pretrained_words = only_include_pretrained_words
        self.tokens_to_add = tokens_to_add
        self.min_pretrained_embeddings = min_pretrained_embeddings


class Pipeline:
    """Manages NLP models configuration and actions.

    Use `Pipeline` for creating new models from a configuration or loading a pre-trained model.
    
    Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.
    
    # Parameters
    
        binary: Optional[str]`
            The path to the model.tar.gz of a pre-trained `Pipeline`
        config: `Optional[PipelineConfiguration]`
            A `PipelineConfiguration` object defining the configuration of the fresh `Pipeline`.
    """
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
    def from_file(
        cls, path: str, vocab_config: Optional[VocabularyConfiguration] = None
    ) -> "Pipeline":
        """Creates a pipeline from a config yaml file path
        
        # Arguments
            path: `str`
                The path to the model.tar.gz of a pre-trained `Pipeline`
            config: `Optional[VocabularyConfiguration]`
                A `PipelineConfiguration` object defining the configuration of a fresh `Pipeline`.
        
        # Returns
            pipeline: `Pipeline`
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

        # Arguments
            config: `Union[str, PipelineConfiguration]`
                A `PipelineConfiguration` object or a YAML `str` for the pipeline configuration
            vocab_config: `Optional[VocabularyConfiguration]`
                A `VocabularyConfiguration` object for associating a vocabulary to the pipeline

        # Returns
            pipeline: `Pipeline`
        """

        if isinstance(config, str):
            config = PipelineConfiguration.from_params(Params(yaml.safe_load(config)))
        pipeline = cls(config=config)

        if vocab_config:
            pipeline = cls._extend_vocab_from_sources(
                pipeline,
                sources=vocab_config.sources,
                max_vocab_size=vocab_config.max_vocab_size,
                min_count=vocab_config.min_count,
                pretrained_files=vocab_config.pretrained_files,
                only_include_pretrained_words=vocab_config.only_include_pretrained_words,
                min_pretrained_embeddings=vocab_config.min_pretrained_embeddings,
                tokens_to_add=vocab_config.tokens_to_add,
            )

        return pipeline
    
    @classmethod
    def from_binary(cls, binary: str, **kwargs) -> "Pipeline":
        """Loads an Allennlp pipeline from binary model.tar.gz file"""
        return cls(binary=binary)

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
    def _allennlp_configuration(self):
        base_config = {
            "config": self.config.as_dict(),
            "type": __default_impl__.__name__,
        }

        return {
            "dataset_reader": base_config,
            "model": base_config,
            "iterator": {
                "batch_size": 32,
                "max_instances_in_memory": 128,
                "sorting_keys": [
                    [
                        self.__forward_inputs()[0],
                        "list_num_tokens"
                        if isinstance(self.model.encoder, TimeDistributedEncoder)
                        else "num_tokens",
                    ]
                ],
                "type": "bucket",
            },
        }
    
    @staticmethod
    def __model_from_config(
        config: PipelineConfiguration, **extra_params
    ) -> __default_impl__:
        """Creates a internal base model from pipeline configuration"""
        return __default_impl__.from_params(Params({"config": config}), **extra_params)
    
    def set_head(self, type: Type[TaskHead], **params):
        """Updates the pipeline head (already created with pipeline model dependency"""
        self._config.head = TaskHeadSpec(type=type.__name__, **params)
        self._model.head = self._config.head.compile(model=self.model)
    
    def train(
        self,
        output: str,
        trainer: str,
        training: str,
        validation: str = None, # TODO: why not optional as the other below?
        test: Optional[str] = None,
        vocab: Optional[str] = None,
        verbose: bool = False,
    ) -> "Pipeline":
        """Launches a training run with the specified parameters and datasets

        # Parameters
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
            vocab: `Optional[str]`
                The already generated vocabulary path
            verbose: `bool`
                Turn on verbose logs
        """
        self._model = self._model.train(mode=True)

        return PipelineHelper.train(
            self,
            TrainConfiguration(
                vocab=vocab,
                test_cfg=test,
                output=output,
                trainer_path=trainer,
                train_cfg=training,
                validation_cfg=validation,
                verbose=verbose,
            ),
        )

    def predict(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        """Applies a data inference"""
        self._model = self._model.eval()
        return self._model.predict(*args, **kwargs)

    def explain(self, *args, **kwargs) -> Dict[str, Any]:
        """Applies a data inference with embedding explanation"""
        return self._model.explain(*args, **kwargs)

    def serve(self, port: int = 9998):
        """Server as rest api"""
        self._model = self._model.eval()
        return PipelineHelper.serve(self, port)

    def explore(
        self,
        ds_path: str,
        explore_id: Optional[str] = None,
        es_host: Optional[str] = None,
        batch_size: int = 500,
        prediction_cache_size: int = 0,
        explain: bool = False,
        force_delete: bool = True,
        **metadata,
    ) -> dd.DataFrame:
        """
            Read a data source and tries to apply a model predictions to the whole data source. The
            results will be persisted into an elasticsearch index for further data exploration
        """
        config = ExploreConfiguration(
            batch_size=batch_size,
            prediction_cache_size=prediction_cache_size,
            explain=explain,
            force_delete=force_delete,
            **metadata,
        )

        es_config = ElasticsearchExplore(
            es_index=explore_id or str(uuid.uuid1()),
            es_host=es_host or constants.DEFAULT_ES_HOST,
        )

        explore_df = PipelineHelper.explore(self, ds_path, config, es_config)
        self._show_explore(es_config)

        return explore_df

    def _show_explore(self, elasticsearch: ElasticsearchExplore) -> None:
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

    def _extend_vocab_from_sources(
        self, sources: List[str], **extra_args
    ) -> "Pipeline":
        """Extends an already created vocabulary from a list of source dictionary"""
        vocab = self._model.vocab
        vocab.extend_from_instances(
            params=Params(extra_args),
            instances=[
                instance
                for data_source in sources
                for instance in self._model.read(data_source)
            ],
        )
        self._model = self.__model_from_config(self.config, vocab=vocab)
        return self

    def __forward_inputs(self) -> List[str]:
        """
        Calculate the required head.forward arguments. We use this method
        for automatically generate data iterator sorting keys
        """
        required, _ = split_signature_params_by_predicate(
            self.head.forward, lambda p: p.default == inspect.Parameter.empty
        )
        return [p.name for p in required] or [None]

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


class TrainConfiguration:
    """Learn configuration class"""

    def __init__(
        self,
        output: str,
        vocab: Optional[str] = None,
        trainer_path: str = "",
        train_cfg: str = "",
        validation_cfg: Optional[str] = None,
        test_cfg: Optional[str] = None,
        verbose: bool = False,
    ):
        self.output = output
        self.vocab = vocab
        self.trainer = trainer_path
        self.training = train_cfg
        self.validation = validation_cfg
        self.test = test_cfg
        self.verbose = verbose


class ExploreConfiguration:
    """Explore configuration data class"""

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


class PipelineHelper:
    """Extra pipeline methods"""

    __LOGGER = logging.getLogger(__name__)

    @classmethod
    def serve(cls, pipeline: Pipeline, port: int):
        """Serves an pipline as rest api"""

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
    def _to_allennlp_configuration(cls, config: TrainConfiguration) -> Dict[str, Any]:
        trainer = {
            **yaml_to_dict(config.trainer),
        }
        trainer["trainer"]["type"] = "default"
        # Add cuda device if necessary
        trainer["trainer"]["cuda_device"] = trainer["trainer"].get(
            "cuda_device", get_env_cuda_device()
        )

        datasets = {}
        for key, path in [
            ("train_data_path", config.training),
            ("validation_data_path", config.validation),
            ("test_data_path", config.test),
        ]:
            if path:
                datasets[key] = path

        return {**trainer, **datasets}

    @classmethod
    def train(cls, pipeline: Pipeline, config: TrainConfiguration):
        logging_level = logging.INFO if config.verbose else logging.WARNING
        logging.getLogger("allennlp").setLevel(logging_level)

        cls.__LOGGER.info("Starting up learning process.")

        fine_tune_params = Params(
            {
                **cls._to_allennlp_configuration(config),
                **pipeline._allennlp_configuration,
            }
        )

        # Force clean folder for run fine tuning properly
        # TODO: reuse vocab
        shutil.rmtree(config.output, ignore_errors=True)
        # _recover_output_folder(config.output, fine_tune_params)

        fine_tune_model(
            model=pipeline._model,
            params=fine_tune_params,
            serialization_dir=config.output,
            extend_vocab=True,  # TODO: Allow parameterize
            file_friendly_logging=True,
        )

        return pipeline.__class__(
            binary=os.path.join(config.output, "model.tar.gz"), config=pipeline.config
        )

    @classmethod
    def explore(
        cls,
        pipeline: Pipeline,
        ds_path: str,
        config: ExploreConfiguration,
        elasticsearch: ElasticsearchExplore,
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
