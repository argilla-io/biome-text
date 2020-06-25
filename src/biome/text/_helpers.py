import copy
import json
import logging
import os
import re
import time
from threading import Thread
from typing import Any, Dict, Optional, Tuple, List
from urllib.error import URLError
from urllib.parse import urlparse

import uvicorn
from allennlp.common import Params
from allennlp.common.util import prepare_environment, sanitize
from allennlp.data import DataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.models import archive_model
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient
from fastapi import FastAPI
from torch.utils.data.dataset import IterableDataset

from biome.text import Pipeline, PipelineConfiguration, TrainerConfiguration, helpers
from biome.text._configuration import ElasticsearchExplore, ExploreConfiguration
from biome.text._model import PipelineModel
from biome.text.constants import EXPLORE_APP_ENDPOINT
from biome.text.data import DataSource, InstancesDataset
from biome.text.errors import http_error_handling
from biome.text.ui import launch_ui

_LOGGER = logging.getLogger(__name__)


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


def _explore(
    pipeline: Pipeline,
    data_source: DataSource,
    config: ExploreConfiguration,
    elasticsearch: ElasticsearchExplore,
) -> dd.DataFrame:
    """
    Executes a pipeline prediction over a datasource and register results int a elasticsearch index

    Parameters
    ----------
    pipeline
    data_source
    config
    elasticsearch

    Returns
    -------

    """
    if config.prediction_cache > 0:
        pipeline.init_prediction_cache(config.prediction_cache)

    ddf_mapped = data_source.to_mapped_dataframe()
    # Stringify input data for better elasticsearch index mapping integration,
    # avoiding properties with multiple value types (string and long,...)
    for column in ddf_mapped.columns:
        ddf_mapped[column] = ddf_mapped[column].apply(helpers.stringify)

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
    # Keep as metadata only non used values/columns
    ddf_source = ddf_source[
        [c for c in ddf_source.columns if c not in ddf_mapped.columns]
    ]
    ddf_mapped["metadata"] = ddf_source.map_partitions(
        lambda df: helpers.stringify(sanitize(df.to_dict(orient="records")))
    )

    ddf = DaskElasticClient(
        host=elasticsearch.es_host, retry_on_timeout=True, http_compress=True
    ).save(ddf_mapped, index=elasticsearch.es_index, doc_type=elasticsearch.es_doc)

    elasticsearch.create_explore_data_index(force_delete=config.force_delete)
    elasticsearch.create_explore_data_record(
        {
            **(config.metadata or {}),
            "datasource": data_source.source,
            # TODO: This should change when ui is normalized (action detail and action link naming)
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

    def launch_ui_app(ui_port: int) -> Thread:
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

    waiting_seconds = 1
    url = f"{EXPLORE_APP_ENDPOINT}/{elasticsearch.es_index}"

    if not is_service_up(url):
        port = urlparse(EXPLORE_APP_ENDPOINT).port
        if not port:
            _LOGGER.warning(
                "Cannot start explore application. "
                "Please, be sure you can reach %s from your browser "
                "or configure 'BIOME_EXPLORE_ENDPOINT' environment variable",
                url,
            )
            return
        launch_ui_app(port)
    time.sleep(waiting_seconds)
    _LOGGER.info("You can access to your data exploration from this url: %s", url)
    show_func = (
        show_notebook_explore
        if helpers.is_running_on_notebook()
        else show_browser_explore
    )
    show_func(url)


class PipelineTrainer:
    """
    Default trainer for `PipelineModel`

    Parameters
    ----------
    pipeline:
        The trainable pipeline
    trainer_config:
        The trainer configuration
    output_dir:
        The used training folder
    training:
        The training instances dataset
    validation:
        The validation instances dataset. Optional
    test:
        The test instances dataset. Optional
    batch_weight_key:
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    embedding_sources_mapping:
        mapping from model paths to the pretrained embedding filepaths
        used during fine-tuning.
    epoch_callbacks:
        A list of callbacks that will be called at the end of every epoch, and at the start of
        training (with epoch = -1).
    """

    __LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        pipeline: Pipeline,
        trainer_config: TrainerConfiguration,
        output_dir: str,
        training: InstancesDataset,
        validation: Optional[InstancesDataset] = None,
        test: Optional[InstancesDataset] = None,
        batch_weight_key: str = "",
        embedding_sources_mapping: Dict[str, str] = None,
        epoch_callbacks: List["allennlp.training.EpochCallback"] = None,
    ):
        self._pipeline = pipeline
        self._trainer_config = copy.deepcopy(trainer_config)
        self._output_dir = output_dir
        self._batch_weight_key = batch_weight_key
        self._embedding_sources_mapping = embedding_sources_mapping
        self._batch_size = self._trainer_config.batch_size
        self._training = training
        self._validation = validation
        self._test = test
        self._epoch_callbacks = epoch_callbacks

        self._setup()

    def _setup(self):
        """Setup the trainer components and local resources"""

        prepare_environment(Params({}))
        os.makedirs(self._output_dir, exist_ok=True)

        # We don't need to load pretrained weights from saved models
        if self._pipeline.config.features.word:
            self._pipeline.config.features.word.weights_file = None

        serialization_params = sanitize(
            self._allennlp_configuration(self._pipeline.config, self._trainer_config)
        )
        with open(os.path.join(self._output_dir, CONFIG_NAME), "w") as param_file:
            json.dump(serialization_params, param_file, indent=4)

        self._pipeline.save_vocabulary(os.path.join(self._output_dir, "vocabulary"))

        for dataset in [self._training, self._validation, self._test]:
            if dataset and hasattr(dataset, "index_with"):
                dataset.index_with(self._pipeline.backbone.vocab)

        trainer_params = Params(
            self._trainer_as_allennlp_configuration(self._trainer_config)
        )

        no_grad_regexes = trainer_params.pop(
            "no_grad", ()
        )  # This could be nice to have exposed

        pipeline_model = self._pipeline._model
        for name, parameter in pipeline_model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        training_data_loader = self._configure_dataloader(
            self._training,
            batch_size=self._trainer_config.batch_size,
            with_data_bucketing=self._trainer_config.data_bucketing,
        )

        validation_data_loader = (
            self._configure_dataloader(
                self._validation,
                batch_size=self._trainer_config.batch_size,
                with_data_bucketing=self._trainer_config.data_bucketing,
            )
            if self._validation
            else None
        )

        self._trainer = Trainer.from_params(
            model=pipeline_model,
            serialization_dir=self._output_dir,
            data_loader=training_data_loader,
            validation_data_loader=validation_data_loader,
            params=trainer_params,
            epoch_callbacks=self._epoch_callbacks,
        )

    @staticmethod
    def _configure_dataloader(
        dataset: InstancesDataset, batch_size: int, with_data_bucketing: bool
    ):
        """
        Configures a pytorch dataloader for a given dataset, with a batch_size and setting
        data bucketing if enabled and is possible.
        """
        return (
            DataLoader(
                dataset,
                batch_sampler=BucketBatchSampler(
                    data_source=dataset, batch_size=batch_size
                ),
            )
            if with_data_bucketing and not isinstance(dataset, IterableDataset)
            else DataLoader(dataset, batch_size=batch_size)
        )

    def test_evaluation(self) -> Dict[str, Any]:
        """
        Evaluates the model against the test dataset (if defined)

        Returns
        -------
        Test metrics information

        """
        test_data = self._test
        if not test_data:
            return {}

        self.__LOGGER.info("The model will be evaluated using the best epoch weights.")
        return evaluate(
            self._pipeline._model,
            data_loader=DataLoader(test_data, batch_size=self._batch_size),
            cuda_device=self._trainer.cuda_device,
            batch_weight_key=self._batch_weight_key,
        )

    def train(self) -> Tuple[str, Dict[str, Any]]:
        """
        Train the inner model with given configuration on initialization

        Returns
        -------
        A tuple with trained model path and related metrics information
        """

        from allennlp.models.model import _DEFAULT_WEIGHTS

        try:
            metrics = self._trainer.train()
        except KeyboardInterrupt:
            # if we have completed an epoch, try to create a model archive.
            if os.path.exists(os.path.join(self._output_dir, _DEFAULT_WEIGHTS)):
                logging.info(
                    "Fine-tuning interrupted by the user. Attempting to create "
                    "a model archive using the current best epoch weights."
                )
                self.save_best_model()
            raise

        for k, v in self.test_evaluation().items():
            metrics["test_" + k] = v

        self.save_best_model()

        with open(os.path.join(self._output_dir, "metrics.json"), "w") as metrics_file:
            metrics_json = json.dumps(metrics, indent=2)
            metrics_file.write(metrics_json)

        return os.path.join(self._output_dir, "model.tar.gz"), metrics

    def save_best_model(self):
        """Packages the best model as tar.gz archive"""
        archive_model(self._output_dir)

    @staticmethod
    def _trainer_as_allennlp_configuration(
        trainer: TrainerConfiguration,
    ) -> Dict[str, Any]:
        """Creates trainer configuration dict"""
        __excluded_keys = ["data_bucketing", "batch_size"]  # Data loader attributes
        trainer_config = {
            k: v for k, v in vars(trainer).items() if k not in __excluded_keys
        }
        trainer_config.update(
            {
                "checkpointer": {"num_serialized_models_to_keep": 1},
                "tensorboard_writer": {"should_log_learning_rate": True},
            }
        )
        return trainer_config

    @classmethod
    def _allennlp_configuration(
        cls,
        pipeline_config: PipelineConfiguration,
        trainer_config: TrainerConfiguration,
    ) -> Dict[str, Any]:
        """Creates a allennlp configuration for pipeline train experiment configuration"""

        allennlp_config = {
            "trainer": cls._trainer_as_allennlp_configuration(trainer_config),
            "model": {
                "config": pipeline_config.as_dict(),
                "type": PipelineModel.__name__,
            },
        }

        return copy.deepcopy({k: v for k, v in allennlp_config.items() if v})
