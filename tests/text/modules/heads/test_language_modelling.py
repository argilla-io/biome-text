from typing import Dict

import pandas as pd
import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration


@pytest.fixture
def training_dataset() -> Dataset:
    """Creating the dataframe."""
    data = {
        "text": [
            "this is a text",
            "my name is dani",
            "this is a table",
            "my name is paco",
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def pipeline_dict() -> Dict:
    """Creating the pipeline dictionary"""

    pipeline_dict = {
        "name": "lm",
        "features": {
            "word": {"embedding_dim": 50, "lowercase_tokens": True, "trainable": True},
            "char": {
                "embedding_dim": 50,
                "dropout": 0.1,
                "encoder": {
                    "type": "gru",
                    "hidden_size": 10,
                    "num_layers": 1,
                    "bidirectional": True,
                },
            },
        },
        "encoder": {
            "type": "gru",
            "num_layers": 1,
            "hidden_size": 10,
            "bidirectional": True,
        },
        "head": {"type": "LanguageModelling", "dropout": 0.1, "bidirectional": True},
    }

    return pipeline_dict


@pytest.fixture
def trainer_dict() -> Dict:
    """Creating the trainer dictionary"""

    trainer_dict = {
        "num_epochs": 10,
        "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.002},
    }

    return trainer_dict


def test_train(pipeline_dict, training_dataset, trainer_dict, tmp_path):
    """Testing the correct working of prediction, vocab creating and training"""

    pipeline = Pipeline.from_config(pipeline_dict)
    pipeline.predict(text="my name is juan")

    pipeline.train(
        output=str(tmp_path / "lm"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_dataset,
        validation=training_dataset,
    )
