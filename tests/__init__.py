import logging
import os

logging.basicConfig(level=logging.INFO)
test_logger = logging.getLogger(__name__)

TESTS_BASEPATH = os.path.dirname(__file__)
RESOURCES_PATH = os.path.join(TESTS_BASEPATH, "resources")
TUTORIALS_PATH = os.path.join(
    TESTS_BASEPATH, "..", "docs", "docs", "documentation", "tutorials"
)

from .test_support import *
