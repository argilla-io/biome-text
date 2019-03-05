from typing import Optional

from allennlp.data import DatasetReader
from allennlp.models import Archive
from allennlp.predictors import Predictor

from biome.allennlp.predictors import DefaultBasePredictor
import logging

_logger = logging.getLogger(__name__)


def get_predictor_from_archive(
    archive: Archive, predictor_name: Optional[str] = None
) -> Predictor:
    # Matching predictor name with model name
    model_config = archive.config.get("model")
    dataset_reader_config = archive.config.get("dataset_reader")
    predictor = predictor_name if predictor_name else model_config.get("type")
    try:
        return Predictor.from_archive(archive, predictor)
    except Exception as e:
        _logger.warning("Cannot create predictor {}, usind default. Error: {}".format(predictor, e))
        ds_reader = DatasetReader.from_params(dataset_reader_config)
        return DefaultBasePredictor(archive.model, ds_reader)
