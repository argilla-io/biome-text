from numpy.testing import assert_allclose

from biome.text import Pipeline


def test_classification_output():
    labels = ["a", "b", "c", "d", "e", "f"]
    pipeline = Pipeline.from_config(
        {
            "name": "test_text_classification",
            "head": {"type": "TextClassification", "labels": labels},
        }
    )
    prediction = pipeline.predict(text="test")

    assert prediction.keys() == dict(labels=None, probabilities=None).keys()
    assert len(prediction["labels"]) == len(prediction["probabilities"]) == 6
    assert_allclose(
        sorted(prediction["probabilities"], reverse=True), prediction["probabilities"]
    )
    assert all([isinstance(label, str) for label in prediction["labels"]])
    assert set(labels) == set(prediction["labels"])
    assert all([isinstance(prob, float) for prob in prediction["probabilities"]])
