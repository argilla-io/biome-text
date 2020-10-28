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
            "entities": [
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
        "features": {"word": {"embedding_dim": 2}},
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
        "cuda_device": -1,
    }

    return trainer_dict


def test_tokenization_with_blank_tokens(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)
    predictions = pipeline.predict(text="Test this text \n \n", entities=[])
    assert len(predictions["tags"][0]) == 4


def test_default_explain(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)

    prediction = pipeline.explain("This is a simple text")
    assert prediction["explain"]
    assert len(prediction["explain"]["text"]) == len(prediction["tags"][0])
    # enable training mode for generate instances with tags
    pipeline.head.train()

    prediction = pipeline.explain(text="This is a simple text", entities=[])
    assert len(prediction["explain"]["tags"]) == len(prediction["explain"]["text"])

    for label in prediction["explain"]["tags"]:
        assert "label" in label
        assert "token" in label


def test_train(pipeline_dict, training_data_source, trainer_dict, tmp_path):
    pipeline = Pipeline.from_config(pipeline_dict)

    assert pipeline.output == ["entities", "tags"]

    assert pipeline.head.span_labels == ["NER"]
    assert pipeline.head.labels == ["B-NER", "I-NER", "U-NER", "L-NER", "O"]

    predictions = pipeline.predict(["test", "this", "pretokenized", "text"])
    assert "entities" in predictions
    assert "tags" in predictions
    for entity in predictions["entities"][0]:
        assert "start_token" in entity
        assert "end_token" in entity
        assert "label" in entity
        assert "start" not in entity
        assert "end" not in entity

    predictions = pipeline.predict_batch(
        [{"text": "Test this NER system"}, {"text": "and this"}]
    )
    assert "entities" in predictions[0]
    assert "tags" in predictions[0]

    for entity in predictions[0]["entities"][0]:
        assert "start" in entity
        assert "end" in entity

    pipeline.create_vocabulary(VocabularyConfiguration(sources=[training_data_source]))

    pipeline.train(
        output=str(tmp_path / "ner_experiment"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_data_source,
        validation=training_data_source,
    )
