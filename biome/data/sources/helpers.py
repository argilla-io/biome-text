import logging
import os
from copy import deepcopy

from allennlp.common.file_utils import cached_path
from dask.bag import Bag
from typing import Dict, Optional

from biome.allennlp.data.transformations import biome_datasource_spec_to_dataset_config
from biome.data.helpers import is_elasticsearch_configuration
from biome.data.sources import JSON_FORMAT
from biome.data.sources.elasticsearch import from_elasticsearch
from biome.data.sources.example_preparator import ExamplePreparator, RESERVED_FIELD_PREFIX
from biome.data.sources.file import from_json, from_csv

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def __transform_example(data: Dict, example_preparator: ExamplePreparator) -> Optional[Dict]:
    try:
        return example_preparator.read_info(data)
    except Exception as ex:
        __logger.warning(ex)
        return None


def is_reserved_field(field: str) -> bool:
    return field and str(field).startswith(RESERVED_FIELD_PREFIX)


def read_dataset(cfg: Dict, include_source: bool = False) -> Bag:
    try:
        config = biome_datasource_spec_to_dataset_config(deepcopy(cfg))
    except Exception as e:
        __logger.warning(e)
        config = cfg

    example_preparator = ExamplePreparator(config.pop('transformations', {}), include_source)

    return __build_dataset(config) \
        .map(__transform_example, example_preparator) \
        .filter(lambda example: example is not None) \
        .filter(lambda example: example['tokens'] is not None) \  # CHAPUZA!!!
        .persist()
    # TODO: We have to think about a more general solution for the CHAPUZA,
    #       once we have more model types and get rid of the forward in the model yml!

def __build_dataset(config: Dict) -> Bag:
    params = {k: v for k, v in config.items()}  # Preserve original config (multiple reads)

    if is_elasticsearch_configuration(params):
        return from_elasticsearch(**params)

    path: str = params.pop('path')
    path = path if path.endswith('*') else cached_path(path)
    if not 'format' in params:
        _, extension = os.path.splitext(path)
        params['format'] = extension[1:]

    format: str = params.pop('format', JSON_FORMAT)
    if __is_json(format):
        return from_json(path, **params)
    else:
        return from_csv(path, **params, assume_missing=True)


def __is_json(format: str) -> bool:
    return JSON_FORMAT in format
