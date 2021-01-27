import pytest
from numpy.testing import assert_allclose

from biome.text import Pipeline
from biome.text.modules.heads.task_prediction import Attribution


@pytest.fixture
def pipeline() -> Pipeline:
    labels = ["a", "b", "c", "d", "e", "f"]
    return Pipeline.from_config(
        {
            "name": "test_text_classification",
            "head": {"type": "TextClassification", "labels": labels},
        }
    )


def test_classification_output(pipeline):
    prediction = pipeline.predict(text="test")

    assert prediction.keys() == dict(labels=None, probabilities=None).keys()
    assert len(prediction["labels"]) == len(prediction["probabilities"]) == 6
    assert_allclose(
        sorted(prediction["probabilities"], reverse=True), prediction["probabilities"]
    )
    assert all([isinstance(label, str) for label in prediction["labels"]])
    assert set(pipeline.head.labels) == set(prediction["labels"])
    assert all([isinstance(prob, float) for prob in prediction["probabilities"]])


def test_attributions(pipeline):
    instance = pipeline.head.featurize("test this sentence")
    forward_output = pipeline._model.forward_on_instances([instance])

    attributions = pipeline.head._compute_attributions(forward_output[0], instance)

    assert all([isinstance(attribution, Attribution) for attribution in attributions])
    assert len(attributions) == 3
    assert all(attr.field == "text" for attr in attributions)
    assert all([isinstance(attr.attribution, float) for attr in attributions])
    assert attributions[1].start == 5 and attributions[1].end == 9
