import logging
from typing import Dict, Iterable, Tuple

from biome.data.utils import get_nested_property_from_data
from dask.bag import Bag
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ID_FIELD = '@id'
__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def __to_es_document(data: Dict, index: str, type: str, id_field: str) -> Dict:
    document = {
        '_index': index,
        '_type': type,
        '_source': data
    }

    if id_field:
        document['_id'] = get_nested_property_from_data(data, id_field)

    return document


def __bulk_data(data: Iterable[Dict], es_hosts: str, es_batch_size: int) -> Tuple:
    es = __es_client(es_hosts)
    return bulk(es, actions=data, stats_only=True, chunk_size=es_batch_size)


def __es_client(es_hosts):
    return Elasticsearch(hosts=es_hosts, retry_on_timeout=True)


def __prepare_index(index: str, type: str, es_hosts: str):
    es = __es_client(es_hosts)

    dynamic_templates = [
        {data_type: {
            "match_mapping_type": data_type,
            "path_match": path_match,
            "mapping": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            }
        }} for data_type, path_match in [
            ('*', '*.value'),
            ('string', '*')
        ]
    ]

    es.indices.delete(index=index, ignore=[400, 404])
    es.indices.create(index=index, body={
        "mappings": {
            type: {
                "dynamic_templates": dynamic_templates
            }
        }
    }, ignore=400)
    pass


def es_sink(dataset: Bag,
            index: str,
            type: str,
            es_hosts: str,
            es_batch_size: int = 1000,
            id_field: str = ID_FIELD) -> Iterable[Tuple]:
    __prepare_index(index, type, es_hosts)

    return dataset \
        .map(__to_es_document, index=index, type=type, id_field=id_field) \
        .map_partitions(__bulk_data, es_batch_size=es_batch_size, es_hosts=es_hosts)
