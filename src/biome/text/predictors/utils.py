import logging
from typing import Optional

from allennlp.common.checks import ConfigurationError
from allennlp.models import Archive
from allennlp.predictors import Predictor

from biome.text.dataset_readers.datasource_reader import DataSourceReader
from biome.text.predictors import DefaultBasePredictor

__LOGGER = logging.getLogger(__name__)


def get_predictor_from_archive(
    archive: Archive, predictor_name: Optional[str] = None
) -> Predictor:
    """Loads a model predictor from a model.tar.gz file"""

    # Matching predictor name with model name
    model_config = archive.config.get("model")
    dataset_reader_config = archive.config.get("dataset_reader")
    predictor_name = predictor_name or model_config.get("type")
    try:
        return Predictor.from_archive(archive, predictor_name)
    except ConfigurationError as error:
        # If there is no corresponding predictor to the model, we use the DefaultBasePredictor
        __LOGGER.warning("%s; Using the 'DefaultBasePredictor'!", error)
        ds_reader = DataSourceReader.from_params(dataset_reader_config)
        return DefaultBasePredictor(archive.model, ds_reader)
