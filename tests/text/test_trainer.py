import os
import tempfile
from typing import Tuple

import pytest
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text.configuration import TrainerConfiguration


@pytest.fixture
def train_valid_dataset(resources_data_path) -> Tuple[Dataset, Dataset]:
    """Returns both training and validation datasets"""

    training_ds = Dataset.from_csv(
        paths=str(resources_data_path / "business.cat.2k.train.csv")
    )
    validation_ds = Dataset.from_csv(
        paths=str(resources_data_path / "business.cat.2k.valid.csv")
    )

    return training_ds, validation_ds


@pytest.fixture
def pipeline_dict() -> dict:
    return {
        "name": "german_business_names",
        "features": {
            "word": {"embedding_dim": 16, "lowercase_tokens": True},
            "char": {
                "embedding_dim": 16,
                "encoder": {
                    "type": "gru",
                    "num_layers": 1,
                    "hidden_size": 32,
                    "bidirectional": True,
                },
                "dropout": 0.1,
            },
        },
        "head": {
            "type": "TextClassification",
            "labels": [
                "Unternehmensberatungen",
                "Friseure",
                "Tiefbau",
                "Dienstleistungen",
                "Gebrauchtwagen",
                "Restaurants",
                "Architekturbüros",
                "Elektriker",
                "Vereine",
                "Versicherungsvermittler",
                "Sanitärinstallationen",
                "Edv",
                "Maler",
                "Physiotherapie",
                "Werbeagenturen",
                "Apotheken",
                "Vermittlungen",
                "Hotels",
                "Autowerkstätten",
                "Elektrotechnik",
                "Allgemeinärzte",
                "Handelsvermittler Und -vertreter",
            ],
            "pooler": {
                "type": "gru",
                "num_layers": 1,
                "hidden_size": 16,
                "bidirectional": True,
            },
            "feedforward": {
                "num_layers": 1,
                "hidden_dims": [16],
                "activations": ["relu"],
                "dropout": [0.1],
            },
        },
    }


def test_default_root_dir(
    change_to_tmp_working_dir, pipeline_dict, train_valid_dataset
):
    pl = Pipeline.from_config(pipeline_dict)
    trainer = Trainer(pl, train_dataset=train_valid_dataset[0])
    assert trainer.trainer.default_root_dir == str(
        change_to_tmp_working_dir / "training_logs"
    )


def test_deep_copy_of_trainer_config(pipeline_dict, train_valid_dataset):
    pl = Pipeline.from_config(pipeline_dict)
    trainer_config = TrainerConfiguration()
    trainer = Trainer(
        pl, train_dataset=train_valid_dataset[0], trainer_config=trainer_config
    )
    assert trainer_config is not trainer._trainer_config


@pytest.mark.parametrize(
    "input_kwargs,expected_loggers",
    [
        ({}, ["csv", "tensorboard", "wandb"]),
        ({"logger": False}, []),
        (
            {
                "logger": MLFlowLogger(
                    tracking_uri=os.path.join(tempfile.gettempdir(), "mlruns")
                ),
                "add_wandb_logger": False,
            },
            ["csv", "tensorboard", "mlflow"],
        ),
        (
            {
                "logger": [
                    MLFlowLogger(
                        tracking_uri=os.path.join(tempfile.gettempdir(), "mlruns")
                    ),
                    CSVLogger(save_dir=tempfile.gettempdir()),
                ],
                "add_wandb_logger": False,
                "add_tensorboard_logger": False,
            },
            ["csv", "mlflow"],
        ),
    ],
)
def test_add_default_loggers(
    input_kwargs, expected_loggers, pipeline_dict, train_valid_dataset, tmp_path
):
    trainer_config = TrainerConfiguration(
        **input_kwargs, default_root_dir=str(tmp_path)
    )
    trainer = Trainer(
        Pipeline.from_config(pipeline_dict),
        train_dataset=train_valid_dataset[0],
        trainer_config=trainer_config,
    )
    if input_kwargs.get("logger") is not False:
        assert isinstance(trainer.trainer.logger, LoggerCollection)
        assert len(trainer.trainer.logger.experiment) == len(expected_loggers)
    else:
        assert trainer._trainer_config.logger is False

    def loggers_include(logger_type) -> bool:
        return any(
            [
                isinstance(logger, logger_type)
                for logger in trainer._trainer_config.logger
            ]
        )

    for logger in expected_loggers:
        if logger == "csv":
            assert loggers_include(CSVLogger)
        if logger == "tensorboard":
            assert loggers_include(TensorBoardLogger)
        if logger == "wandb":
            assert loggers_include(WandbLogger)
            assert (tmp_path / "wandb").is_dir()
        if logger == "mlflow":
            assert loggers_include(MLFlowLogger)


def test_pipeline_test(pipeline_dict, train_valid_dataset, tmp_path):
    import json

    pl = Pipeline.from_config(pipeline_dict)
    trainer = Trainer(pl)
    first_metrics = trainer.test(
        train_valid_dataset[1], output_dir=tmp_path, batch_size=16
    )
    assert "test_loss" in first_metrics

    assert (tmp_path / "metrics.json").is_file()
    with (tmp_path / "metrics.json").open() as file:
        assert "test_loss" in json.load(file)

    assert pl.evaluate(train_valid_dataset[1])["test_loss"] == pytest.approx(
        first_metrics["test_loss"]
    )
