import warnings
from warnings import warn_explicit

import pkg_resources

from .pipeline import (
    Pipeline,
    PipelineConfiguration,
    TrainerConfiguration,
    VocabularyConfiguration,
)

warnings.showwarning = warn_explicit

try:
    __version__ = pkg_resources.get_distribution(__name__.replace(".", "-")).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
