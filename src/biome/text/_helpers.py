import copy
import inspect
import time
from threading import Thread
from typing import Any, Dict, List
from urllib.error import URLError

import uvicorn
from allennlp.common.util import sanitize
from biome.text import Pipeline, TrainerConfiguration, helpers
from biome.text._configuration import (
    ElasticsearchExplore,
    ExploreConfiguration,
    TrainConfiguration,
    _ModelImpl,
)

from biome.text.data import DataSource
from biome.text.errors import http_error_handling
from biome.text.modules.encoders import TimeDistributedEncoder
from biome.text.ui import launch_ui
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient
from fastapi import FastAPI


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

    def iterator_configuration(
        pipeline: Pipeline, trainer: TrainerConfiguration
    ) -> Dict[str, Any]:
        """Creates a data iterator configuration"""

        def _forward_inputs() -> List[str]:
            """
            Calculate the required head.forward arguments. We use this method
            for automatically generate data iterator sorting keys
            """
            required, _ = helpers.split_signature_params_by_predicate(
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
                                pipeline.backbone.encoder, TimeDistributedEncoder
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
        "type": _ModelImpl.__name__,
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
        f"http://localhost:{ui_port}/{elasticsearch.es_index}"
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
