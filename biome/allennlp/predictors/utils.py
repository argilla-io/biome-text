from allennlp.data import DatasetReader
from allennlp.models import Archive, Model
from allennlp.predictors import Predictor

from biome.allennlp.predictors import DefaultBasePredictor


def get_predictor_from_archive(archive: Archive) -> Predictor:
    # Matching predictor name with model name
    model_config = archive.config.get("model")
    dataset_reader_config = archive.config.get('dataset_reader')
    model_type = model_config.get("type")
    try:
        return Predictor.from_archive(archive, model_type)
    except Exception:
        ds_reader = DatasetReader.from_params(dataset_reader_config)
        return DefaultBasePredictor(archive.model, ds_reader)
