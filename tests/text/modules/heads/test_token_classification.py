from typing import Dict

import pytest
from biome.text import Pipeline, VocabularyConfiguration, TrainerConfiguration
from biome.text.data import DataSource
import pandas as pd


@pytest.fixture
def training_data_source(tmp_path) -> DataSource:
    data_file = tmp_path / "train.json"
    df = pd.DataFrame(
        {
            "text": [
                "This is a simple NER test",
                "This is a simple NER test with misaligned spans",
                "No NER here",
            ],
            "labels": [
                [{"start": 17, "end": 20, "label": "NER"}],
                [{"start": 17, "end": 22, "label": "NER"}],
                [],
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
        "name": "biome-bimpm",
        "features": {"word": {"embedding_dim": 2,},},
        "head": {
            "type": "TokenClassification",
            "labels": ["NER"],
            "label_encoding": "BIOUL",
        },
    }

    return pipeline_dict


@pytest.fixture
def trainer_dict() -> Dict:
    trainer_dict = {
        "num_epochs": 1,
        "batch_size": 1,
        "optimizer": {"type": "adam", "lr": 0.01},
    }

    return trainer_dict


def test_train(pipeline_dict, training_data_source, trainer_dict, tmp_path):
    pipeline = Pipeline.from_config(pipeline_dict)
    pipeline.predict(text="Test this NER machine")
    pipeline.create_vocabulary(VocabularyConfiguration(sources=[training_data_source]))

    pipeline.train(
        output=str(tmp_path / "ner_experiment"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_data_source,
        validation=training_data_source,
    )
