import datetime
import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Dict, List, Optional, Union
from urllib.error import URLError
from urllib.parse import urlparse

import datasets
import elasticsearch
import elasticsearch.helpers
import pandas as pd
from allennlp.common.util import sanitize
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient
from elasticsearch import Elasticsearch

from biome.text import Pipeline, constants, helpers
from biome.text.data import DataSource
from biome.text.dataset import Dataset
from biome.text.ui import launch_ui

_LOGGER = logging.getLogger(__name__)


@dataclass
class DataExploration:
    """
    Data exploration info
    """

    name: str
    datasource_name: str
    pipeline_name: str
    pipeline_type: str
    task_name: str
    use_prediction: bool
    datasource_columns: List[str] = field(default_factory=list)
    pipeline_inputs: List[str] = field(default_factory=list)
    pipeline_outputs: List[str] = field(default_factory=list)
    pipeline_labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_old_format(self) -> Dict[str, Any]:
        """

        Returns
        -------

        """
        signature = [self.pipeline_inputs + self.pipeline_outputs]
        return {
            **self.metadata,
            "datasource": self.datasource_name,
            # TODO: This should change when ui is normalized (action detail and action link naming)
            "explore_name": self.name,
            "model": self.pipeline_name,
            "columns": self.datasource_columns,
            "metadata_columns": [
                c for c in self.datasource_columns if c not in signature
            ],
            "pipeline": self.pipeline_type,
            "output": self.pipeline_outputs,
            "inputs": self.pipeline_inputs,  # backward compatibility
            "signature": signature,
            "predict_signature": self.pipeline_inputs,
            "labels": self.pipeline_labels,
            "task": self.task_name,
            "use_prediction": True,  # TODO(frascuchon): flag for ui backward compatibility. Remove in the future
        }


class _ExploreOptions:
    """Configures an exploration run

    Parameters
    ----------
    batch_size
        DataSource: The batch size for indexing predictions (default is `500)
        Dataset: Batch size for the predictions
    num_proc
        Only for Dataset: Number of processes to run predictions in parallel (default: 1)
    prediction_cache_size
        The size of the cache for caching predictions (default is `0)
    explain
        Whether to extract and return explanations of token importance (default is `False`)
    force_delete
        Whether to delete existing explore with `explore_id` before indexing new items (default is `True)
    **metadata
        Additional metadata to index in Elasticsearch
    """

    def __init__(
        self,
        batch_size: int = 500,
        num_proc: int = 1,
        prediction_cache_size: int = 0,
        explain: bool = False,
        force_delete: bool = True,
        **metadata,
    ):
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.prediction_cache = prediction_cache_size
        self.explain = explain
        self.force_delete = force_delete
        self.metadata = metadata


class _ElasticsearchDAO:
    """Elasticsearch data exploration class"""

    def __init__(self, es_host: Optional[str] = None):
        self.es_host = es_host or constants.DEFAULT_ES_HOST
        if not self.es_host.startswith("http"):
            self.es_host = f"http://{self.es_host}"

        self.client = Elasticsearch(
            hosts=self.es_host, retry_on_timeout=True, http_compress=True
        )
        self.es_doc = helpers.get_compatible_doc_type(self.client)

    def create_explore_index(
        self,
        es_index: str,
        data_exploration: DataExploration,
        ddf_content: Union[dd.DataFrame, datasets.Dataset],
        force_delete: bool,
    ) -> None:
        """Creates an exploration data record data exploration"""
        self.__create_data_index(es_index, force_delete)

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
            id=es_index,
            body={
                "doc": dict(
                    name=es_index,
                    created_at=datetime.datetime.now(),
                    **data_exploration.as_old_format(),
                ),
                "doc_as_upsert": True,
            },
        )

        if isinstance(ddf_content, datasets.Dataset):
            column_names = ddf_content.column_names
            ddf_content = ddf_content.map(
                lambda x: {
                    col_name: helpers.stringify(x[col_name]) for col_name in column_names
                }
            )
            self._index_dataset(dataset=ddf_content, index=es_index)
        else:
            # Stringify input data for better elasticsearch index mapping integration,
            # avoiding properties with multiple value types (string and long,...)
            for column in ddf_content.columns:
                ddf_content[column] = ddf_content[column].apply(helpers.stringify)

            (
                DaskElasticClient(
                    host=self.es_host, retry_on_timeout=True, http_compress=True
                )
                .save(ddf_content, index=es_index, doc_type=self.es_doc)
                .persist()
            )

    def __create_data_index(self, es_index: str, force_delete: bool):
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
            self.client.indices.delete(index=es_index, ignore=[400, 404])

        self.client.indices.create(
            index=es_index,
            body={"mappings": {self.es_doc: {"dynamic_templates": dynamic_templates}}},
            ignore=400,
            params={"include_type_name": "true"},
        )

    def _index_dataset(self, dataset: datasets.Dataset, index: str):
        number_of_docs = len(dataset)
        successes = 0

        def passage_generator():
            for idx, example in enumerate(dataset):
                yield {"_id": idx, **example}

        # create the ES index
        for ok, action in elasticsearch.helpers.streaming_bulk(
            client=self.client, index=index, actions=passage_generator()
        ):
            successes += ok
        if successes != number_of_docs:
            _LOGGER.warning(
                f"Some documents failed to be added to ElasticSearch. Failures: {number_of_docs - successes}/{number_of_docs}"
            )
        _LOGGER.info("Indexed %d documents" % (successes,))


def create(
    pipeline: Pipeline,
    data_source: [DataSource, Dataset],
    explore_id: Optional[str] = None,
    es_host: Optional[str] = None,
    batch_size: int = 50,
    num_proc: int = 1,
    prediction_cache_size: int = 0,
    explain: bool = False,
    force_delete: bool = True,
    show_explore: bool = True,
    **metadata,
) -> str:
    """Launches the Explore UI for a given data source

    Running this method inside an `IPython` notebook will try to render the UI directly in the notebook.

    Running this outside a notebook will try to launch the standalone web application.

    Parameters
    ----------
    pipeline
        Pipeline used for data exploration
    data_source
        The data source/data set
    explore_id
        A name or id for this explore run, useful for running and keep track of several explorations
    es_host
        The URL to the Elasticsearch host for indexing predictions (default is `localhost:9200`)
    batch_size
        DataSource: The batch size for indexing predictions (default is `500)
        Dataset: Batch size for the predictions
    num_proc
        Only for Dataset: Number of processes to run predictions in parallel (default: 1)
    prediction_cache_size
        The size of the cache for caching predictions (default is `0)
    explain
        Whether to extract and return explanations of token importance (default is `False`)
    force_delete
        Deletes exploration with the same `explore_id` before indexing the new explore items (default is `True)
    show_explore
        If true, show ui for data exploration interaction (default is `True`)
    """

    opts = _ExploreOptions(
        batch_size=batch_size,
        num_proc=num_proc,
        prediction_cache_size=prediction_cache_size,
        explain=explain,
        force_delete=force_delete,
        **metadata,
    )

    explore_id = explore_id or str(uuid.uuid1())

    es_dao = _ElasticsearchDAO(es_host=es_host)

    if isinstance(data_source, Dataset):
        _explore_a_dataset(
            explore_id=explore_id,
            pipeline=pipeline,
            dataset=data_source.dataset,
            options=opts,
            es_dao=es_dao,
        )
    else:
        data_source.mapping = pipeline._update_ds_mapping_with_pipeline_input_output(
            data_source
        )

        _explore(
            explore_id,
            pipeline=pipeline,
            data_source=data_source,
            options=opts,
            es_dao=es_dao,
        )

    if show_explore:
        show(explore_id, es_host=es_host)
    return explore_id


# TODO (dcfidalgo): Explore operation should be rewrite in terms of predict_batch + evaluate methods when ready
def _explore(
    explore_id: str,
    pipeline: Pipeline,
    data_source: DataSource,
    options: _ExploreOptions,
    es_dao: _ElasticsearchDAO,
) -> None:
    """
    Executes a pipeline prediction over a datasource and register results int a elasticsearch index

    Parameters
    ----------
    pipeline
    data_source
    options
    es_dao

    Returns
    -------

    """
    if options.prediction_cache > 0:
        pipeline.init_prediction_cache(options.prediction_cache)

    ddf_mapped = data_source.to_mapped_dataframe()

    # this only makes really sense when we have a predict_batch_json method implemented ...
    n_partitions = max(1, round(len(ddf_mapped) / options.batch_size))

    apply_func = pipeline.explain_batch if options.explain else pipeline.predict_batch

    def add_prediction(df: pd.DataFrame) -> pd.Series:
        """Runs and returns the predictions for a given input dataframe"""
        input_batch = df.to_dict(orient="records")
        predictions = apply_func(input_batch)
        return pd.Series(map(sanitize, predictions), index=df.index)

    # a persist is necessary here, otherwise it fails for n_partitions == 1
    # the reason is that with only 1 partition we pass on a generator to predict_batch_json
    ddf_mapped: dd.DataFrame = ddf_mapped.repartition(
        npartitions=n_partitions
    ).persist()
    ddf_mapped["prediction"] = ddf_mapped.map_partitions(
        add_prediction, meta=(None, object)
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

    data_exploration = DataExploration(
        name=explore_id,
        datasource_name=data_source.source,
        datasource_columns=data_source.to_dataframe().columns.values.tolist(),
        pipeline_name=pipeline.name,
        pipeline_type=pipeline.type_name,
        pipeline_inputs=pipeline.inputs,
        pipeline_outputs=pipeline.output,
        pipeline_labels=pipeline.head.labels,  # TODO(dvilasuero,dcfidalgo): Only for text classification ??????
        task_name=pipeline.head.task_name().as_string(),
        use_prediction=True,
        metadata=options.metadata or {},
    )
    es_dao.create_explore_index(
        es_index=explore_id,
        data_exploration=data_exploration,
        ddf_content=ddf_mapped,
        force_delete=options.force_delete,
    )


def _explore_a_dataset(
    explore_id: str,
    pipeline: Pipeline,
    dataset: datasets.Dataset,
    options: _ExploreOptions,
    es_dao: _ElasticsearchDAO,
):
    if options.prediction_cache > 0:
        pipeline.init_prediction_cache(options.prediction_cache)

    # TODO: Here we should use a future evaluate method that takes as required input also the labels!!!
    # Maybe a predict should actually fail if you pass on a label ...
    apply_func = pipeline.explain_batch if options.explain else pipeline.predict_batch

    def add_predictions(batch, columns):
        # For the last batch, this batch_size can be smaller than the batch_size specified in the map function!
        batch_size = len(batch[pipeline.inputs[0]])
        input_dicts = [
            {
                col_name: batch[col_name][i]
                for col_name, optional in columns
                if batch.get(col_name)
            }
            for i in range(batch_size)
        ]
        predictions = apply_func(input_dicts)
        return {"prediction": sanitize(predictions)}

    # we include the pipeline.output as input columns so we do not use it for the metadata
    meta_columns = [
        col_name
        for col_name in dataset.column_names
        if col_name not in pipeline.inputs + pipeline.output
    ]
    dataset = dataset.map(
        lambda x: {"metadata": {col_name: x[col_name] for col_name in meta_columns}},
        remove_columns=meta_columns,
    ).map(
        add_predictions,
        fn_kwargs={
            "columns": [(col, False) for col in pipeline.inputs]
            + [(col, True) for col in pipeline.output]
        },
        batched=True,
        batch_size=options.batch_size,
        num_proc=options.num_proc,
    )

    data_exploration = DataExploration(
        name=explore_id,
        datasource_name=list(dataset.info.download_checksums.keys())[
            0
        ],  # TODO: find a better way ...
        datasource_columns=meta_columns,
        pipeline_name=pipeline.name,
        pipeline_type=pipeline.type_name,
        pipeline_inputs=pipeline.inputs,
        pipeline_outputs=pipeline.output,
        pipeline_labels=pipeline.head.labels,  # TODO(dvilasuero,dcfidalgo): Only for text classification ??????
        task_name=pipeline.head.task_name().as_string(),
        use_prediction=True,
        metadata=options.metadata or {},
    )

    es_dao.create_explore_index(
        es_index=explore_id,
        data_exploration=data_exploration,
        ddf_content=dataset,
        force_delete=options.force_delete,
    )


def show(explore_id: str, es_host: Optional[str] = None) -> None:
    """Shows explore ui for data prediction exploration"""

    es_dao = _ElasticsearchDAO(es_host=es_host)

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
            kwargs=dict(es_host=es_dao.es_host, port=ui_port),
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
    url = f"{constants.EXPLORE_APP_ENDPOINT}/{explore_id}"

    if not is_service_up(url):
        port = urlparse(constants.EXPLORE_APP_ENDPOINT).port
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
