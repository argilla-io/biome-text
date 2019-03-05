import json
import logging

from dask.bag import Bag

from biome.data.sinks.elasticsearch import es_sink
from biome.data.sinks.file import file_sink

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import warnings
from typing import Dict

warnings.simplefilter(action="ignore", category=FutureWarning)


def is_elasticsearch_configuration(config: Dict):
    return "index" in config and "es_hosts" in config


def store_dataset(dataset: Bag, store_config: Dict):
    if is_elasticsearch_configuration(store_config):
        return es_sink(dataset, **store_config)
    else:
        return file_sink(dataset.map(json.dumps), store_config)
