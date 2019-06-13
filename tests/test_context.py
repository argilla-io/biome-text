import json
import os
import tempfile
from typing import Dict

import logging

logging.basicConfig(level=logging.DEBUG)

TEST_RESOURCES = os.path.dirname(__file__)

os.chdir(TEST_RESOURCES)


def create_temp_configuration(data: Dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as config_file:
        config_file.writelines(json.dumps(data).split("\n"))
        config_file.close()
        return config_file.name
