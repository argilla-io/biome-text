import warnings
from warnings import warn_explicit

import pkg_resources

warnings.showwarning = warn_explicit

try:
    __version__ = pkg_resources.get_distribution(__name__.replace(".", "-")).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass

# This is necessary, since the from_param machinery needs our classes to be registered!
from . import dataset_readers, models, predictors, pipelines

from biome.text.pipelines.pipeline import Pipeline
