import datetime
import logging
import time
import uuid
from cmath import isclose
from dataclasses import dataclass
from dataclasses import field
from threading import Thread
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlparse

import elasticsearch.helpers
from allennlp.common.util import sanitize
from elasticsearch import Elasticsearch

from biome.text import Pipeline
from biome.text import constants
from biome.text import helpers
from biome.text.dataset import Dataset
from biome.text.modules.heads import TaskName
from biome.text.ui import launch_ui

_LOGGER = logging.getLogger(__name__)


@dataclass
class DataExploration:
    """
    Data exploration info
    """

    name: str
    pipeline: Pipeline
    use_prediction: bool
    dataset_name: str
    dataset_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_old_format(self) -> Dict[str, Any]:
        """

        Returns
        -------

        """
        signature = [self.pipeline.inputs + self.pipeline.output]
        return {
            **self.metadata,
            "datasource": self.dataset_name,
            # TODO: This should change when ui is normalized (action detail and action link naming)
            "explore_name": self.name,
            "model": self.pipeline.name,
            "columns": self.dataset_columns,
            "metadata_columns": [c for c in self.dataset_columns if c not in signature],
            "pipeline": self.pipeline.type_name,
            "pipeline_config": self.pipeline.config.as_dict(),
            "output": self.pipeline.output,
            "inputs": self.pipeline.inputs,  # backward compatibility
            "signature": signature,
            "predict_signature": self.pipeline.inputs,
            "labels": self.pipeline.head.labels,
            "task": self.pipeline.head.task_name.as_string(),
            "use_prediction": True,  # TODO(frascuchon): flag for ui backward compatibility. Remove in the future
        }


class _ExploreOptions:
    """Configures an exploration run

    Parameters
    ----------
    batch_size
        Batch size for the predictions
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
        dataset: Dataset,
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

        # we have total control over `prediction` field, so we don't need normalize it
        column_names = [c for c in dataset.column_names if c != "prediction"]
        dataset = dataset.map(
            lambda x: {
                col_name: helpers.stringify(x[col_name]) for col_name in column_names
            }
        )
        self._index_dataset(dataset=dataset, index=es_index)

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

    def _index_dataset(self, dataset: Dataset, index: str):
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
    dataset: Dataset,
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
    dataset
        The dataset to explore
    explore_id
        A name or id for this explore run, useful for running and keep track of several explorations
    es_host
        The URL to the Elasticsearch host for indexing predictions (default is `localhost:9200`)
    batch_size
        Batch size for the predictions
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

    _explore(
        explore_id=explore_id,
        pipeline=pipeline,
        dataset=dataset,
        options=opts,
        es_dao=es_dao,
    )

    if show_explore:
        show(explore_id, es_host=es_host)
    return explore_id


def _explore(
    explore_id: str,
    pipeline: Pipeline,
    dataset: Dataset,
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

        return {
            "prediction": _make_prediction_backward_compatible(sanitize(predictions))
        }

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

    # Quick fix for in-memory data that are not backed up by a file
    # TODO: Find a better solution, maybe introduce a Dataset.name attribute
    try:
        dataset_name = list(dataset.dataset.info.download_checksums.keys())[0]
    except AttributeError:
        dataset_name = "InMemory"

    data_exploration = DataExploration(
        name=explore_id,
        pipeline=pipeline,
        use_prediction=True,
        dataset_name=dataset_name,
        dataset_columns=meta_columns,
        metadata=options.metadata or {},
    )

    es_dao.create_explore_index(
        es_index=explore_id,
        data_exploration=data_exploration,
        dataset=dataset,
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
        from IPython.core.display import HTML
        from IPython.core.display import display

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


def _make_prediction_backward_compatible(predictions: List[Dict]):
    """Makes the Classification prediction output backward compatible with the UI"""
    # check if predictions are from a Classification TaskHead
    if not all([key in predictions[0].keys() for key in ["labels", "probabilities"]]):
        return predictions

    # a little trick to know if this is a multilable classification ...
    is_multilabel = isclose(sum(predictions[0]["probabilities"]), 1.0)

    for prediction in predictions:
        prediction["classes"] = {
            label: prob
            for label, prob in zip(prediction["labels"], prediction["probabilities"])
        }
        prediction["probs"] = prediction["probabilities"]
        # we don't have the logits anymore, so just duplicate the probs ...
        prediction["logits"] = prediction["probabilities"]

        if not is_multilabel:
            prediction["label"] = prediction["labels"][0]
            prediction["prob"] = prediction["probabilities"][0]
            prediction["max_class"] = prediction["labels"][0]  # deprecated
            prediction["max_class_prob"] = prediction["probabilities"][0]  # deprecated

    return predictions
