import pkg_resources

# This is necessary, since the from_param machinery needs our classes to be registered!
from . import dataset_readers, models, predictors

try:
    __version__ = pkg_resources.get_distribution(__name__.replace(".", "-")).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
