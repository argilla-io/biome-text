import os

import pandas as pd
import pytest
from ray import tune

from biome.text import Pipeline
from biome.text.data import DataSource
from biome.text.hpo import HpoExperiment, HpoParams


@pytest.fixture
def datasource_test(tmp_path) -> DataSource:
    data_file = os.path.abspath(tmp_path / "classifier.parquet")
    df = pd.DataFrame(
        {
            "text": ["A common text", "This is why you get", "Seriosly?, I'm not sure"],
            "label": ["Yes", "No", "Other"],
        }
    )
    df.to_parquet(data_file)

    return DataSource(source=str(data_file))


def test_hpo_experiment_generation(datasource_test: DataSource):

    pipeline = Pipeline.from_config(
        {
            "name": "test",
            "tokenizer": {"lang": "en"},
            "features": {"word": {"embedding_dim": 5, "lowercase_tokens": True}},
            "head": {"type": "TextClassification", "labels": ["Yes", "No", "Other"]},
        }
    )

    experiment = HpoExperiment(
        name="test-experiment",
        pipeline=pipeline,
        train=datasource_test.source,
        validation=datasource_test.source,
        num_samples=1,
        hpo_params=HpoParams(
            pipeline={
                "features": {
                    "word": {"embedding_dim": tune.choice([32, 64, 128])},
                    "char": {
                        "embedding_dim": 8,
                        "lowercase_characters": True,
                        "encoder": {
                            "type": "gru",
                            "hidden_size": tune.choice([32, 64, 128]),
                            "bidirectional": True,
                        },
                        "dropout": tune.uniform(0, 0.5),
                    },
                },
                "encoder": {
                    "type": "lstm",
                    "num_layers": 1,
                    "hidden_size": tune.choice([128, 256, 512]),
                    "dropout": tune.uniform(0, 0.5),
                    "bidirectional": True,
                },
            },
            trainer={
                "optimizer": {"type": "adam", "lr": tune.loguniform(0.001, 0.01)},
                "num_epochs": 10,
                "batch_size": tune.choice([1, 2, 3]),
            },
        ),
    )

    config = experiment.as_tune_experiment().spec["config"]
    assert config["name"] == "test-experiment"
    assert config["train"] == datasource_test.source
    assert config["validation"] == datasource_test.source
    assert config["trainer"]["num_epochs"] == 10
    assert config["pipeline"]["features"]["word"]["lowercase_tokens"]
    assert config["pipeline"]["features"]["char"]["embedding_dim"] == 8

    analysis = tune.run(experiment.as_tune_experiment())
    assert len(analysis.trials) == experiment.num_samples
