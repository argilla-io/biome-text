import datetime
from typing import Any, Dict, Optional

import allennlp
from elasticsearch import Elasticsearch

from biome.text import constants
from biome.text._model import PipelineModel
from . import helpers


def __register(impl_class, overrides: bool = False):
    """Register the impl. class in allennlp components registry"""

    allennlp.models.Model.register(impl_class.__name__, exist_ok=overrides)(impl_class)
    allennlp.data.DatasetReader.register(impl_class.__name__, exist_ok=overrides)(
        impl_class
    )


__register(PipelineModel, overrides=True)


class ExploreConfiguration:
    """Configures an exploration run

    Parameters
    ----------
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


class ElasticsearchExplore:
    """Elasticsearch data exploration class"""

    def __init__(self, es_index: str, es_host: Optional[str] = None):
        self.es_index = es_index
        self.es_host = es_host or constants.DEFAULT_ES_HOST
        if not self.es_host.startswith("http"):
            self.es_host = f"http://{self.es_host}"

        self.client = Elasticsearch(
            hosts=self.es_host, retry_on_timeout=True, http_compress=True
        )
        self.es_doc = helpers.get_compatible_doc_type(self.client)

    def create_explore_data_record(self, parameters: Dict[str, Any]):
        """Creates an exploration data record data exploration"""

        self.client.indices.create(
            index=constants.BIOME_METADATA_INDEX,
            body={
                "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0,}}
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
            params={"include_type_name": "true"},
        )
