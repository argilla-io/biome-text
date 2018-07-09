import warnings
from typing import Dict

warnings.simplefilter(action='ignore', category=FutureWarning)


def is_elasticsearch_configuration(config: Dict):
    return "index" in config and "es_hosts" in config


from recognai.data import dataset_readers
