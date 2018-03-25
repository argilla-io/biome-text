import json
import logging
from typing import Dict, Any

from allennlp.common import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CsvConfig(object):
    __DEFAULT_DELIMITER = ','

    def __init__(self, delimiter: str = __DEFAULT_DELIMITER):
        self._delimiter = delimiter

    @classmethod
    def from_params(cls, params: Params) -> 'CsvConfig':

        if isinstance(params, str):
            return CsvConfig()
        else:
            delimiter = params.pop('delimiter', CsvConfig.__DEFAULT_DELIMITER)
            return CsvConfig(delimiter)


def is_json(format: Any) -> bool:
    format_type = 'json' if not format \
        else str(format).lower() if type(format) is str \
        else str(format['type']).lower() if type(format) is dict \
        else None

    return 'json' == format_type


def ds_format(params):
    return params.pop('dataset_format', None)
