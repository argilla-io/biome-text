import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("biome-text").version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass

import logging

# configure basic 'biome.text' logging
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(_handler)
_LOGGER.setLevel("INFO")

from .configuration import LightningTrainerConfiguration
from .configuration import PipelineConfiguration
from .configuration import TrainerConfiguration
from .configuration import VocabularyConfiguration
from .dataset import Dataset
from .pipeline import Pipeline
from .trainer import Trainer
