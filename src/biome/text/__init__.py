import warnings
from warnings import warn_explicit

import pkg_resources

from biome.text.pipelines import Pipeline

warnings.showwarning = warn_explicit

try:
    __version__ = pkg_resources.get_distribution(__name__.replace(".", "-")).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
