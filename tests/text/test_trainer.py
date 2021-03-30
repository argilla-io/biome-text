import os
import random
import tempfile
from typing import Tuple

import numpy as np
import pytest
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import VocabularyConfiguration
from biome.text.configuration import CharFeatures
from biome.text.configuration import LightningTrainerConfiguration
from biome.text.configuration import WordFeatures


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


def test_text_classification(tmp_path, pipeline_dict, train_valid_dataset):
    """Apart from a well specified training, this also tests the vocab creation!"""

    random.seed(42)
    np.random.seed(422)
    torch.manual_seed(4222)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4222)

    pl = Pipeline.from_config(pipeline_dict)
    train_ds = train_valid_dataset[0]
    valid_ds = train_valid_dataset[1]

    vocab_config = VocabularyConfiguration(max_vocab_size={"word": 50})
    trainer_config = LightningTrainerConfiguration(
        batch_size=64,
        optimizer={"type": "adam", "lr": 0.01},
        max_epochs=5,
        default_root_dir=str(tmp_path),
    )

    trainer = Trainer(
        pipeline=pl,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        trainer_config=trainer_config,
        vocab_config=vocab_config,
    )

    trainer.fit()

    assert pl.vocab.get_vocab_size(WordFeatures.namespace) == 52
    assert pl.vocab.get_vocab_size(CharFeatures.namespace) == 83

    assert pl.num_trainable_parameters == 22070

    evaluation = pl.evaluate(valid_ds)

    assert evaluation["loss"] == pytest.approx(0.8217981046438217, abs=0.003)


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
    trainer_config = LightningTrainerConfiguration()
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
    trainer_config = LightningTrainerConfiguration(
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
