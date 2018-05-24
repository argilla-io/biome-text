import logging
from typing import Any, Dict

from allennlp.common import Params

from recognai.data.readers import JSON_FORMAT

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def is_elasticsearch_source(config: Dict):
    return "index" in config and "client_kwargs" in config


def is_json(format: str) -> bool:
    return JSON_FORMAT in format
