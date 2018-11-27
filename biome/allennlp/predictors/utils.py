from allennlp.models import Archive
from allennlp.predictors import Predictor


def get_predictor_from_archive(archive: Archive) -> Predictor:
    # Matching predictor name with model name
    model_type = archive.config.get("model").get("type")
    return Predictor.from_archive(archive, model_type)
