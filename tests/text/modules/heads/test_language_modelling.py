from typing import Dict

import pandas as pd
import pytest
import yaml
from biome.text import TrainerConfiguration, VocabularyConfiguration
from biome.text import Pipeline
from biome.text.data import DataSource


@pytest.fixture
def training_data_source(tmp_path) -> DataSource:
    data_file = tmp_path / "record_pairs.json"
    df = pd.DataFrame(
        {
            "text": [
                "this is a text",
                "my name is dani",
                "this is a table",
                "my name is paco",
            ],
        }
    )
    df.to_json(data_file, lines=True, orient="records")

    return DataSource(
        source=str(data_file), flatten=False, lines=True, orient="records"
    )


@pytest.fixture
def pipeline_dict() -> Dict:
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
        "head": {"type": "LanguageModelling", "dropout": 0.1, "bidirectional": True},
    }

    return pipeline_dict


@pytest.fixture
def trainer_dict() -> Dict:
    trainer_dict = {
        "num_epochs": 10,
        "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.002},
    }

    return trainer_dict


def test_train(pipeline_dict, training_data_source, trainer_dict, tmp_path):
    pipeline = Pipeline.from_config(pipeline_dict)
    pipeline.predict(text="my name is juan")
    pipeline.create_vocabulary(VocabularyConfiguration(sources=[training_data_source]))

    pipeline.train(
        output=str(tmp_path / "lm"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_data_source,
        validation=training_data_source,
    )
