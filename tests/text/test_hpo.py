import os
import pickle

import pandas as pd
import pytest
import mlflow
from ray import tune

from biome.text import Pipeline, VocabularyConfiguration
from biome.text.data import DataSource
from biome.text.dataset import Dataset
from biome.text.hpo import HpoExperiment, HpoParams, TuneExperiment


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


@pytest.fixture
def dataset():
    return Dataset.from_dict({"text": ["a", "b"], "label": ["a", "b"]})


@pytest.fixture
def pipeline_config():
    return {
        "name": "test_ray_tune_trainable",
        "features": {"word": {"embedding_dim": 2},},
        "head": {"type": "TextClassification", "labels": ["a", "b"]},
    }


@pytest.fixture
def trainer_config() -> dict:
    return {
        "optimizer": {"type": "adam", "lr": 0.01},
        "num_epochs": 1,
        "batch_size": 2,
    }


def test_tune_exp_default_trainable(dataset, pipeline_config, trainer_config, monkeypatch):
    # avoid logging to wandb
    monkeypatch.setenv("WANDB_MODE", "dryrun")

    pipeline_config["features"]["word"]["embedding_dim"] = tune.choice([2, 4])
    trainer_config["optimizer"]["lr"] = tune.loguniform(0.001, 0.01)

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
        num_samples=1,
    )

    assert my_exp._name.startswith("HPO on")
    assert my_exp.name == my_exp._name
    assert my_exp._run_identifier == "_default_trainable"

    analysis = tune.run(my_exp)
    assert len(analysis.trials) == 1

    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    assert mlflow.get_experiment_by_name(my_exp._name)


def test_tune_exp_save_dataset_and_vocab(dataset, pipeline_config, trainer_config, monkeypatch):
    pl = Pipeline.from_config(pipeline_config)
    pl.create_vocabulary(VocabularyConfiguration(sources=[dataset]))

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
        vocab=pl.backbone.vocab,
    )

    config = my_exp.config
    pl2 = Pipeline.from_config(config["pipeline_config"], config["vocab_path"])

    assert pl.backbone.vocab._index_to_token == pl2.backbone.vocab._index_to_token
    assert pl.backbone.vocab._token_to_index == pl2.backbone.vocab._token_to_index

    assert dataset[:] == Dataset.load_from_disk(config["train_dataset_path"])[:]
    assert dataset[:] == Dataset.load_from_disk(config["valid_dataset_path"])[:]


def test_tune_exp_custom_trainable(dataset, pipeline_config, trainer_config, monkeypatch):
    def my_trainable(config):
        pass

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
        name="custom trainable",
        trainable=my_trainable
    )

    assert my_exp.name == "custom trainable"
    assert my_exp.trainable == my_trainable
    assert my_exp._run_identifier == "my_trainable"
