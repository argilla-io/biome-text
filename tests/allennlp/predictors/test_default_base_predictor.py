import os

import pytest
from allennlp.models import load_archive
from biome.allennlp.predictors import get_predictor_from_archive

from tests.test_context import TEST_RESOURCES

MODEL_PATH = os.path.join(
    TEST_RESOURCES, "resources/models/eng_es_word_classifier/model.tar.gz"
)


@pytest.fixture
def predictor():
    model_archive = load_archive(MODEL_PATH)
    # if predictor_name not registered, it will use the DefaultBasePredictor
    return get_predictor_from_archive(
        model_archive, predictor_name="non_existent_predictor"
    )


def test_predict_json(predictor):
    inputs = [
        {"tokens": "horse"},
        {"tokens": "horse", "gold_label": "check", "further_garbage": 43},
        {"tokens": ""},
    ]

    for json_input in inputs[:2]:
        results = predictor.predict_json(json_input)
        for key in ["input", "annotation"]:
            assert key in results.keys()

        # These asserts should maybe go to the SequenceClassifier model test, there we define the output ...
        for key in ["logits", "classes", "max_class", "max_class_prob"]:
            assert key in results["annotation"].keys()

        assert results["annotation"]["max_class"] == "english"
        assert results["annotation"]["max_class_prob"] == pytest.approx(0.7124, 0.0001)

    with pytest.raises(ValueError) as err:
        predictor.predict_json(inputs[2])
    assert "could not be converted to an instance. No prediction possible." in str(err)


def test_predict_batch_json(predictor):
    inputs = [
        {"tokens": "horse"},
        {"tokens": "horse", "gold_label": "check", "further_garbage": 43},
        {"tokens": ""},
    ]

    results = predictor.predict_batch_json(inputs)
    assert len(results) == len(inputs) - 1

    with pytest.raises(IndexError) as err:
        predictor.predict_batch_json([inputs[2]])
    assert "No instances found in batch. Check input or make batch size bigger." in str(
        err
    )
