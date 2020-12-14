import mlflow
import pytest
from ray import tune

from biome.text import Pipeline
from biome.text import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.hpo import TuneExperiment


@pytest.fixture
def dataset():
    return Dataset.from_dict({"text": ["a", "b"], "label": ["a", "b"]})


@pytest.fixture
def pipeline_config():
    return {
        "name": "test_ray_tune_trainable",
        "features": {
            "word": {"embedding_dim": 2},
        },
        "head": {"type": "TextClassification", "labels": ["a", "b"]},
    }


@pytest.fixture
def trainer_config() -> dict:
    return {
        "optimizer": {"type": "adam", "lr": 0.01},
        "num_epochs": 1,
        "batch_size": 2,
    }


def test_tune_exp_default_trainable(
    dataset, pipeline_config, trainer_config, monkeypatch
):
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


def test_tune_exp_save_dataset_and_vocab(
    dataset, pipeline_config, trainer_config, monkeypatch
):
    pl = Pipeline.from_config(pipeline_config)
    vocab = VocabularyConfiguration(datasets=[dataset]).build_vocab(pipeline=pl)

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
        vocab=vocab,
    )

    config = my_exp.config
    pl2 = Pipeline.from_config(config["pipeline_config"], config["vocab_path"])

    pl._model.extend_vocabulary(vocab)
    assert pl.backbone.vocab._index_to_token == pl2.backbone.vocab._index_to_token
    assert pl.backbone.vocab._token_to_index == pl2.backbone.vocab._token_to_index

    assert dataset[:] == Dataset.load_from_disk(config["train_dataset_path"])[:]
    assert dataset[:] == Dataset.load_from_disk(config["valid_dataset_path"])[:]


def test_tune_exp_custom_trainable(
    dataset,
    pipeline_config,
    trainer_config,
):
    def my_trainable(config):
        pass

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
        name="custom trainable",
        trainable=my_trainable,
    )

    assert my_exp.name == "custom trainable"
    assert my_exp.trainable == my_trainable
    assert my_exp._run_identifier == "my_trainable"
