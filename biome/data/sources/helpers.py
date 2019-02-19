import logging
import os
from typing import Dict, Optional

from biome.data.sources import file
from biome.data.sources.elasticsearch import from_elasticsearch
from biome.data.sources.example_preparator import ExamplePreparator
from dask.bag import Bag

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def __transform_example(data: Dict, example_preparator: ExamplePreparator) -> Optional[Dict]:
    try:
        return example_preparator.read_info(data)
    except Exception as ex:
        __logger.warning(ex)
        return None


def read_dataset(config: Dict, include_source: bool = False) -> Bag:
    example_preparator = ExamplePreparator(config.pop('forward', {}), include_source)

    return __build_dataset(config) \
        .map(__transform_example, example_preparator) \
        .filter(lambda example: example is not None)


def format_from_params(path: str, params) -> Optional[str]:
    format_field_name = 'format'

    if path and format_field_name not in params:
        _, extension = os.path.splitext(path)
        params[format_field_name] = extension[1:]

    return params.pop(format_field_name).lower() if params.get(format_field_name) else None


def __build_dataset(config: Dict) -> Bag:
    params = {k: v for k, v in config.items()}  # Preserve original config (multiple reads)

    supported_formats = {
        'xls': (file.from_excel, dict(na_filter=False, keep_default_na=False, dtype=str)),
        'xlsx': (file.from_excel, dict(na_filter=False, keep_default_na=False, dtype=str)),
        'csv': (file.from_csv, dict(assume_missing=False, na_filter=False, dtype=str)),
        'json': (file.from_json, dict()),
        'jsonl': (file.from_json, dict()),
        'json-l': (file.from_json, dict()),
        'raw': (file.from_documents, dict(recursive=True)),
        'document': (file.from_documents, dict(recursive=True)),
        'pdf': (file.from_documents, dict(recursive=True)),
        'elasticsearch': (from_elasticsearch, dict()),
        'elastic': (from_elasticsearch, dict()),
        'es': (from_elasticsearch, dict())
    }

    format = format_from_params(params.get('path'), params)

    if format in supported_formats:
        dataset_reader, extra_arguments = supported_formats[format]
        return dataset_reader(**{**params, **extra_arguments})
    else:
        raise Exception(
            'Format {} not supported. Supported formats are: {}'.format(format, ' '.join(supported_formats))
        )
