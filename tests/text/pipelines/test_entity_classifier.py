from biome.text.pipelines.entity_classifier import EntityClassifierPipeline
from biome.text.dataset_readers.entity_classifier_reader import EntityClassifierReader
from inspect import signature


def test_predict_signature():
    predict_signature = signature(EntityClassifierPipeline.predict)
    reader_signature = signature(EntityClassifierReader.text_to_instance)

    for key, value in predict_signature.parameters.items():
        assert key in reader_signature.parameters
        assert reader_signature.parameters[key].annotation == value.annotation
