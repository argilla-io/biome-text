from pathlib import Path

import pytest
from ray import tune

from biome.text import Pipeline
from biome.text import TrainerConfiguration
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
def trainer_config() -> TrainerConfiguration:
    return TrainerConfiguration(
        max_epochs=1,
        batch_size=2,
        add_wandb_logger=False,
    )


def test_tune_exp_default_trainable(
    tmp_path, dataset, pipeline_config, trainer_config, monkeypatch
):
    pipeline_config["features"]["word"]["embedding_dim"] = tune.choice([2, 4])
    trainer_config.optimizer["lr"] = tune.loguniform(0.001, 0.01)

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
        num_samples=1,
        local_dir=str(tmp_path),
    )

    assert my_exp._name.startswith("HPO on")
    assert my_exp.name == my_exp._name
    assert my_exp._run_identifier == "_default_trainable"

    analysis = tune.run(my_exp)
    assert len(analysis.trials) == 1


def test_tune_exp_save_dataset_and_vocab(
    dataset, pipeline_config, trainer_config, monkeypatch
):
    pl = Pipeline.from_config(pipeline_config)

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
    )

    config = my_exp.config

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


def test_vocab_config(tmp_path, pipeline_config, trainer_config, dataset):
    vocab_config = VocabularyConfiguration(max_vocab_size=1)

    my_exp = TuneExperiment(
        pipeline_config=pipeline_config,
        trainer_config=trainer_config,
        train_dataset=dataset,
        valid_dataset=dataset,
        vocab_config=vocab_config,
        name="test_vocab_config",
        local_dir=str(tmp_path),
    )

    analysis = tune.run(my_exp)
    pl = Pipeline.from_pretrained(
        Path(analysis.get_best_logdir("validation_loss", "min"))
        / "output"
        / "model.tar.gz"
    )

    assert pl.vocab.get_vocab_size("word") == 3
