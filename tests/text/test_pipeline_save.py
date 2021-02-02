import pytest
from numpy.testing import assert_allclose

from biome.text import Pipeline


@pytest.fixture
def pipeline():
    return Pipeline.from_config(
        {
            "name": "test_pipeline_copy",
            "head": {"type": "TextClassification", "labels": ["a", "b"]},
        }
    )


def test_save(pipeline, tmp_path):
    pipeline.save(tmp_path)

    assert (tmp_path / "model.tar.gz").is_file()

    expected_prediction = pipeline.predict("test")
    prediction = Pipeline.from_pretrained(tmp_path / "model.tar.gz").predict("test")

    assert prediction["labels"] == expected_prediction["labels"]
    assert_allclose(prediction["probabilities"], expected_prediction["probabilities"])
