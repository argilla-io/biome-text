import logging
from typing import Dict, Iterable, Tuple

from dask.bag import Bag
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ID_FIELD = '@id'
__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def __to_es_document(data: Dict, index: str, type: str) -> Dict:
    document = {
        '_index': index,
        '_type': type,
        '_source': data
    }

    if ID_FIELD in data:
        document['_id'] = data[ID_FIELD]

    return document


def __bulk_data(data: Iterable[Dict], es_hosts: str, es_batch_size: int) -> Tuple:
    es = Elasticsearch(hosts=es_hosts, retry_on_timeout=True)
    return bulk(es, actions=data, stats_only=True, chunk_size=es_batch_size)


def es_sink(dataset: Bag, index: str, type: str, es_hosts: str, es_batch_size: int = 1000) -> Iterable[Tuple]:
    return dataset \
        .map(__to_es_document, index=index, type=type) \
        .map_partitions(__bulk_data, es_batch_size=es_batch_size, es_hosts=es_hosts) \
        .compute()
