import logging
import logging
import os
from copy import deepcopy
from typing import Dict

from dask.bag import Bag

from biome.data import is_elasticsearch_configuration
from biome.data.biome.transformations import biome_datasource_spec_to_dataset_config
from biome.data.sources import JSON_FORMAT
from biome.data.sources.elasticsearch import from_elasticsearch
from biome.data.sources.example_preparator import ExamplePreparator, RESERVED_FIELD_PREFIX
from biome.data.sources.file import from_json, from_csv

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def __transform_example(data: Dict, example_preparator: ExamplePreparator) -> Dict:
    try:
        return example_preparator.read_info(data)
    except:
        return None


def is_reserved_field(field: str) -> bool:
    return field and str(field).startswith(RESERVED_FIELD_PREFIX)


def read_dataset(cfg: Dict, include_source: bool = False) -> Bag:

    try:
        config = biome_datasource_spec_to_dataset_config(deepcopy(cfg))
    except:
        config = cfg

    example_preparator = ExamplePreparator(config.pop('transformations', {}), include_source)

    __logger.info("Reading instances from dataset at: %s", config)
    dataset = __build_dataset(config) \
        .map(__transform_example, example_preparator) \
        .persist()
    __logger.info("Finished reading instances")
    return dataset


def __build_dataset(config: Dict) -> Bag:
    params = {k: v for k, v in config.items()}  # Preserve original config (multiple reads)

    if is_elasticsearch_configuration(params):
        return from_elasticsearch(**params)

    if not 'format' in params:
        _, extension = os.path.splitext(params['path'])
        params['format'] = extension[1:]

    format: str = params.pop('format', JSON_FORMAT)
    if __is_json(format):
        return from_json(**params)
    else:
        return from_csv(**params, assume_missing=True)


def __is_json(format: str) -> bool:
    return JSON_FORMAT in format
