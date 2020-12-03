import json

import pytest

from biome.text import Dataset
from biome.text import Pipeline


@pytest.fixture
def dataset() -> Dataset:
    data = {
        "text": ["test", "this", "shaight", "good"],
        "label": ["good", "good", "bad", "good"],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def pipeline(dataset):
    labels = dataset.unique("label")
    return Pipeline.from_config(
        {
            "name": "test_pipeline_evaluate",
            "head": {
                "type": "TextClassification",
                "labels": labels,
            },
        }
    )


def test_pipeline_evaluate(pipeline, dataset, tmp_path):
    prediction_output_file = tmp_path / "prediction_output_file.json"
    metrics = pipeline.evaluate(
        dataset,
        predictions_output_file=str(prediction_output_file),
        batch_size=1,
    )

    assert "loss" in metrics

    predictions = []
    with prediction_output_file.open() as file:
        for line in file.readlines():
            predictions.append(json.loads(line))

    assert len(predictions) == 4
    assert all(["loss" in prediction for prediction in predictions])

    dataset.remove_columns_("label")
    with pytest.raises(ValueError):
        pipeline.evaluate(dataset)
