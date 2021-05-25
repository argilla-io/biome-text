from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_allclose

from biome.text import Pipeline
from biome.text import __version__


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
    assert len(df) == 1 and df["tags.mlflow.runName"][0] == "log_biometext_model"

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
    with (Path(model_uri) / "conda.yaml").open() as file:
        conda_env = yaml.load(file)
        assert conda_env == {
            "name": "mlflow-dev",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.7.9",
                "pip>=20.3.0",
                {"pip": [f"biome-text=={__version__}"]},
            ],
        }
