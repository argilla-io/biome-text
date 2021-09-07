import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("biome-text").version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass

import logging

# configure basic 'biome.text' logging
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(levelname)s:%(name)s: %(message)s")
)  # "%(levelname)s: %(message)s"))
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(_handler)
_LOGGER.setLevel("INFO")
# configure 'allennlp' logging
_ALLENNLP_LOGGER = logging.getLogger("allennlp")
_ALLENNLP_LOGGER.addHandler(_handler)
_ALLENNLP_LOGGER.setLevel("WARNING")

# TODO: Remove this hack when allennlp 1.8.0 is out
import transformers

transformers.__spec__ = ""

from .configuration import PipelineConfiguration
from .configuration import TrainerConfiguration
from .configuration import VocabularyConfiguration
from .dataset import Dataset
from .pipeline import Pipeline
from .trainer import Trainer
