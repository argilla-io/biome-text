import copy
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
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient
from fastapi import FastAPI

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
from . import constants, helpers
from ._pipeline_helper import (
    ElasticsearchExplore,
    ExploreConfiguration,
    TrainConfiguration,
)
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


class Pipeline:
    """Manages NLP models configuration and actions.

    Use `Pipeline` for creating new models from a configuration or loading a pre-trained model.

    Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.
    """

    __LOGGER = logging.getLogger(__name__)
    __TRAINING_CACHE_DATA = "instances_data"

    _model: __default_impl__ = None
    _config: PipelineConfiguration = None

    @classmethod
    def from_yaml(cls, path: str, vocab_path: Optional[str] = None) -> "Pipeline":
        """Creates a pipeline from a config yaml file path

        Parameters
        ----------
        path: `str`
            The path to a YAML configuration file
        vocab_path: `Optional[str]`
            If provided, the pipeline vocab will be loaded from this path

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """
        with open(path) as yamL_file:
            return cls.from_config(yamL_file.read(), vocab_path=vocab_path)

    @classmethod
    def from_config(
        cls, config: Union[str, PipelineConfiguration], vocab_path: Optional[str] = None
    ) -> "Pipeline":
        """Creates a pipeline from a `PipelineConfiguration` object

        Parameters
        ----------
        config: `Union[str, PipelineConfiguration]`
            A `PipelineConfiguration` object or a YAML `str` for the pipeline configuration
        vocab_path: `Optional[str]`
            If provided, the pipeline vocab will be loaded from this path

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """

        if isinstance(config, str):
            config = PipelineConfiguration.from_params(Params(yaml.safe_load(config)))
        return _EmptyPipeline(config=config, vocab=cls._vocab_from_path(vocab_path))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Pipeline":
        """Loads a pipeline from a pre-trained pipeline from a model.tar.gz file path

        Parameters
        ----------
            path: `str`
                The path to the model.tar.gz file of a pre-trained `Pipeline`

        Returns
        -------
            pipeline: `Pipeline`
                A configured pipeline
        """
        return _PreTrainedPipeline(pretrained_path=path, **kwargs)

    def train(
        self,
        output: str,
        trainer: TrainerConfiguration,
        training: str,
        validation: Optional[str] = None,
        test: Optional[str] = None,
        verbose: bool = False,
        extend_vocab: Optional[VocabularyConfiguration] = None,
        restore: bool = True,
    ) -> None:
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
        extend_vocab: `Optional[VocabularyConfiguration]`
            Extends vocab tokens with provided configuration
        restore: `bool`
            If enabled, tries to read previous training status from output folder and
            continues training process from it

        """

        self.__prepare_experiment_folder(output, restore)
        self._model.cache_data(os.path.join(output, self.__TRAINING_CACHE_DATA))

        if extend_vocab:
            self._extend_vocab(vocab_config=extend_vocab)

        # The original pipeline keeps unchanged
        model = copy.deepcopy(self._model)
        config = TrainConfiguration(
            test_cfg=test,
            output=output,
            trainer=trainer,
            train_cfg=training,
            validation_cfg=validation,
            verbose=verbose,
        )

        model.launch_experiment(
            params=Params(_allennlp_configuration(self, config,)),
            serialization_dir=output,
        )

    def __prepare_experiment_folder(self, output: str, restore: bool) -> None:
        """Prepare experiment folder depending of if required experiment restore or not

        Parameters
        ----------
        output
            Path to the output folder
        restore: `bool`
            If False, drops all previous training states

        """
        if not os.path.isdir(output):
            return

        drop_patterns = [
            os.path.join(output, "*.json"),
            os.path.join(output, "**/events.out*"),
        ]

        if not restore:
            drop_patterns.append(os.path.join(output, "*.th"))
            drop_patterns.append(os.path.join(output, self.__TRAINING_CACHE_DATA, "*"))

        for pattern in drop_patterns:
            for file in glob.glob(pattern, recursive=True):
                os.remove(file)

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

        Parameters
        ----------
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

        Returns
        -------
            pipeline: `Pipeline`
                A configured pipeline
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

        explore_df = _explore(self, ds_path, config, es_config)
        _show_explore(es_config)

        return explore_df

    def serve(self, port: int = 9998):
        """Launches a REST prediction service with current model in a specified port (default is `9998)

        # Parameters
            port: `int`
                The port to make available the prediction service
        """
        self._model = self._model.eval()
        return _serve(self, port)

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

    @classmethod
    def _vocab_from_path(cls, from_path: str) -> Optional[Vocabulary]:
        try:
            if not from_path:
                return None
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

    def _update_prediction_signatures(self):
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

    def _load_vocabulary(
        self, vocab_config: VocabularyConfiguration
    ) -> Optional[Vocabulary]:
        """
        Extends a data vocabulary from a given configuration

        Parameters
        ----------
        vocab_config: `VocabularyConfiguration`
            The vocab extension configuration

        Returns
        -------
        vocab: `Optional[Vocabulary]`
            An extended ``Vocabulary`` using the provided configuration

        """

        return self._extend_vocab_from_sources(
            vocab=self.model.vocab,
            sources=vocab_config.sources,
            max_vocab_size=vocab_config.max_vocab_size,
            min_count=vocab_config.min_count,
            pretrained_files=vocab_config.pretrained_files,
            only_include_pretrained_words=vocab_config.only_include_pretrained_words,
            min_pretrained_embeddings=vocab_config.min_pretrained_embeddings,
            tokens_to_add=vocab_config.tokens_to_add,
        )

    def _extend_vocab(self, vocab_config: VocabularyConfiguration) -> None:
        """Extend vocab if no vocab extension was launched before"""
        vocabulary = self._load_vocabulary(vocab_config)
        self._model.update_vocab(vocabulary)
        self._vocab_extended = True


class _EmptyPipeline(Pipeline):
    """
    Parameters
    ----------
        config: `Optional[PipelineConfiguration]`
            A `PipelineConfiguration` object defining the configuration of the fresh `Pipeline`.

    """

    def __init__(self, config: PipelineConfiguration, **extra_args):
        self._config = config
        self._model = self.__model_from_config(self._config, **extra_args)
        if not isinstance(self._model, __default_impl__):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")
        self._update_prediction_signatures()

    @staticmethod
    def __model_from_config(
        config: PipelineConfiguration, **extra_params
    ) -> __default_impl__:
        """Creates a internal base model from pipeline configuration"""
        return __default_impl__.from_params(Params({"config": config}), **extra_params)


class _PreTrainedPipeline(Pipeline):
    """
    Parameters
    ----------

        pretrained_path: `Optional[str]`
            The path to the model.tar.gz of a pre-trained `Pipeline`

    """

    def __init__(self, pretrained_path: str, **extra_args):
        self._binary = pretrained_path
        archive = load_archive(self._binary, **extra_args)
        self._model = self.__model_from_archive(archive)
        self._config = self.__config_from_archive(archive)

        if not isinstance(self._model, __default_impl__):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")
        self._update_prediction_signatures()

    @staticmethod
    def __model_from_archive(archive: Archive) -> __default_impl__:
        if not isinstance(archive.model, __default_impl__):
            raise ValueError(f"Wrong pipeline model: {archive.model}")
        return cast(__default_impl__, archive.model)

    @staticmethod
    def __config_from_archive(archive: Archive) -> PipelineConfiguration:
        config = archive.config["model"]["config"]
        return PipelineConfiguration.from_params(config)

    @property
    def trained_path(self) -> str:
        """Path to binary file when load from binary"""
        return self._binary


def _serve(pipeline: Pipeline, port: int):
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


def _allennlp_configuration(
    pipeline: Pipeline, config: TrainConfiguration
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
        return {k: v for k, v in vars(trainer).items() if k not in __excluded_keys}

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
                pipeline.head.forward, lambda p: p.default == inspect.Parameter.empty,
            )
            return [p.name for p in required] or [None]

        iterator_config = {
            "batch_size": trainer.batch_size,
            "max_instances_in_memory": max(
                trainer.batch_size * trainer.in_memory_batches, trainer.batch_size,
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


def _explore(
    pipeline: Pipeline,
    ds_path: str,
    config: ExploreConfiguration,
    elasticsearch: ElasticsearchExplore,
) -> dd.DataFrame:
    """
    Executes a pipeline prediction over a datasource and register results int a elasticsearch index

    Parameters
    ----------
    pipeline
    ds_path
    config
    elasticsearch

    Returns
    -------

    """
    if config.prediction_cache > 0:
        # TODO: do it
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


def _show_explore(elasticsearch: ElasticsearchExplore) -> None:
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
    url = (
        f"http://localhost:{ui_port}/projects/default/explore/{elasticsearch.es_index}"
    )

    if not is_service_up(url):
        launch_ui_app()

    time.sleep(waiting_seconds)
    show_func = (
        show_notebook_explore
        if helpers.is_running_on_notebook()
        else show_browser_explore
    )
    show_func(url)
