from typing import Dict

import pytest
from biome.text import Pipeline, VocabularyConfiguration, TrainerConfiguration, Dataset
import pandas as pd


@pytest.fixture
def training_dataset() -> Dataset:
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

    return Dataset.from_pandas(df)


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


def test_default_explain(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)

    prediction = pipeline.explain("This is a simple text")
    assert prediction["explain"]
    assert len(prediction["explain"]["text"]) == len(prediction["tags"][0])

    prediction = pipeline.explain(text="This is a simple text", labels=[])
    assert len(prediction["explain"]["labels"]) == len(prediction["explain"]["text"])

    for label in prediction["explain"]["labels"]:
        assert "label" in label
        assert "token" in label


def test_train(pipeline_dict, training_dataset, trainer_dict, tmp_path):
    pipeline = Pipeline.from_config(pipeline_dict)

    assert pipeline.head.span_labels == ["NER"]
    assert pipeline.head.labels == ["B-NER", "I-NER", "U-NER", "L-NER", "O"]

    predictions = pipeline.predict(["test", "this", "pretokenized", "text"])
    assert "entities" not in predictions
    assert "tags" in predictions

    predictions = pipeline.predict_batch(
        [{"text": "Test this NER system"}, {"text": "and this"}]
    )
    assert "entities" in predictions[0]
    assert "tags" in predictions[0]

    pipeline.create_vocabulary(VocabularyConfiguration(sources=[training_dataset]))

    pipeline.train(
        output=str(tmp_path / "ner_experiment"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_dataset,
    )
