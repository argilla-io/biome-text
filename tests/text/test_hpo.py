from ray import tune

from biome.text import Pipeline
from biome.text.hpo import HpoExperiment, HpoParams


def test_hpo_experiment_generation():

    pipeline = Pipeline.from_config(
        {
            "name": "test",
            "tokenizer": {"lang": "en"},
            "features": {
                "word": {
                    "embedding_dim": 5,
                    "lowercase_tokens": True
                }
            },
            "head": {"type": "TextClassification", "labels": ["Yes", "No", "Other"]},
        }
    )

    experiment = HpoExperiment(
        name="test-experiment",
        pipeline=pipeline,
        train="mock",
        validation="mock",
        hpo_params=HpoParams(
            pipeline={
                "features": {
                    "word": {
                        "embedding_dim": tune.choice([32, 64, 128]),
                    },
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
            },
        ),
    )

    config = experiment.as_tune_experiment().spec["config"]
    assert config["name"] == "test-experiment"
    assert config["train"] == "mock"
    assert config["validation"] == "mock"
    assert config["trainer"]["num_epochs"] == 10
    assert config["pipeline"]["features"]["word"]["lowercase_tokens"]
    assert config["pipeline"]["features"]["char"]["embedding_dim"] == 8
