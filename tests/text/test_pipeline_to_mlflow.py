import mlflow
import pandas as pd
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


def test_to_mlflow(pipeline, tmp_path):
    test_str_for_prediction = "test this prediction"
    expected_prediction = pipeline.predict(text=test_str_for_prediction)

    model_uri = pipeline.to_mlflow(
        tracking_uri=str(tmp_path / "to_mlflow_test"), experiment_id=0
    )

    df = mlflow.search_runs(experiment_ids=["0"])
    assert len(df) == 1 and df["tags.mlflow.runName"][0] == "Log biome.text model"

    # load MLFlow model and make predictions
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    prediction: pd.DataFrame = model.predict(
        pd.DataFrame([{"text": test_str_for_prediction}])
    )

    assert len(prediction) == 1
    assert expected_prediction["labels"] == prediction["labels"][0]
    assert_allclose(
        expected_prediction["probabilities"], prediction["probabilities"][0]
    )
