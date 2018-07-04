import logging
import os
from typing import Any, Dict

from allennlp.common import Params
from dask.bag import Bag

from recognai.data import is_elasticsearch_configuration
from recognai.data.biome.transformations import is_biome_datasource_spec, biome_datasource_spec_to_dataset_config
from recognai.data.sources import JSON_FORMAT
from recognai.data.sources.elasticsearch import from_elasticsearch
from recognai.data.sources.example_preparator import ExamplePreparator
from recognai.data.sources.file import from_json, from_csv

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def __transform_example(data: Dict, example_preparator: ExamplePreparator) -> Dict:
    try:
        return example_preparator.read_info(data)
    except:
        return None


def read_dataset(dataset_config: Any) -> Bag:
    if isinstance(dataset_config, str):
        dataset_config = {'path': dataset_config}
    elif isinstance(dataset_config, Params):
        dataset_config = dataset_config.as_dict()

    copy = dataset_config.copy()
    if is_biome_datasource_spec(dataset_config):
        copy = biome_datasource_spec_to_dataset_config(dataset_config)

    example_preparator = ExamplePreparator(copy.pop('transformations', {}))

    __logger.info("Reading instances from dataset at: %s", copy)
    dataset = __build_dataset(copy) \
        .map(__transform_example, example_preparator) \
        .persist()
    __logger.info("Finished reading instances")
    return dataset


def __build_dataset(config: Dict) -> Bag:
    params = config.copy()  # Preserve original config (multiple reads)

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
