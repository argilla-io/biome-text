import logging
from typing import Dict, Iterable, Tuple, Any

from biome.data.utils import get_nested_property_from_data
from dask.bag import Bag
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ID_FIELD = "@id"
__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _to_es_document(data: Any, index: str, type: str, id_field: str) -> Dict:
    resource = data if isinstance(data, Dict) else vars(data)
    document = {"_index": index, "_type": type, "_source": resource}

    if id_field:
        document["_id"] = get_nested_property_from_data(resource, id_field)

    return document


def _bulk_data(data: Iterable[Dict], es_hosts: str, es_batch_size: int) -> Tuple:
    es = _es_client(es_hosts)
    return bulk(es, actions=data, stats_only=True, chunk_size=es_batch_size)


def _es_client(es_hosts):
    return Elasticsearch(hosts=es_hosts, retry_on_timeout=True)


def _prepare_index(index: str, type: str, es_hosts: str):
    es = _es_client(es_hosts)

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

    es.indices.delete(index=index, ignore=[400, 404])
    es.indices.create(
        index=index,
        body={"mappings": {type: {"dynamic_templates": dynamic_templates}}},
        ignore=400,
    )


def es_sink(
    dataset: Bag,
    index: str,
    type: str,
    es_hosts: str,
    es_batch_size: int = 1000,
    id_field: str = None,
    index_recreate: bool = False,
) -> Iterable[Tuple]:
    if index_recreate:
        _prepare_index(index, type, es_hosts)

    return dataset.map(
        _to_es_document, index=index, type=type, id_field=id_field
    ).map_partitions(_bulk_data, es_batch_size=es_batch_size, es_hosts=es_hosts)
