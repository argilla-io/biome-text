import logging
from typing import Dict

from dask.bag import Bag

from recognai.data import is_elasticsearch_configuration
from recognai.data.sinks.elasticsearch import es_sink
from recognai.data.sinks.file import file_sink

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def store_dataset(dataset: Bag, store_config: Dict):
    if is_elasticsearch_configuration(store_config):
        return es_sink(dataset, **store_config)
    else:
        return file_sink(dataset, store_config)
