from typing import Dict

import pandas as pd
import pytest
import spacy

from biome.text import Pipeline, VocabularyConfiguration, TrainerConfiguration, Dataset
from biome.text.helpers import offsets_from_tags


@pytest.fixture
def training_dataset() -> Dataset:
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


def test_tokenization_with_blank_tokens(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)
    predictions = pipeline.predict(text="Test this text \n \n", entities=[])
    assert len(predictions["tags"][0]) == 4


def test_default_explain(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)

    prediction = pipeline.explain("This is a simple text")
    assert prediction["explain"]
    assert len(prediction["explain"]["text"]) == len(prediction["tags"][0][0])
    # enable training mode for generate instances with tags
    pipeline.head.train()

    prediction = pipeline.explain(text="This is a simple text", entities=[])
    assert len(prediction["explain"]["tags"]) == len(prediction["explain"]["text"])

    for label in prediction["explain"]["tags"]:
        assert "label" in label
        assert "token" in label


def test_train(pipeline_dict, training_dataset, trainer_dict, tmp_path):
    # TODO: Some of the asserts are useless because a predict normally does not return any entities!
    #  We should test directly the decode method with a custom TaskOutput!
    pipeline = Pipeline.from_config(pipeline_dict)

    assert pipeline.output == ["entities", "tags"]

    assert pipeline.head.span_labels == ["NER"]
    assert pipeline.head.labels == ["B-NER", "I-NER", "U-NER", "L-NER", "O"]

    predictions = pipeline.predict(["test", "this", "pretokenized", "text"])
    assert predictions.keys() == dict(entities=None, tags=None, scores=None).keys()

    assert isinstance(predictions["entities"], list) and isinstance(predictions["entities"][0], list)
    assert all([isinstance(entity, dict) for entity in predictions["entities"][0]])

    assert isinstance(predictions["tags"], list) and isinstance(predictions["tags"][0], list)
    assert all([isinstance(tag, str) for tag in predictions["tags"][0]])

    pipeline.predict_batch(
        [{"text": "Test this NER system"}, {"text": "and this"}]
    )

    pipeline.create_vocabulary(VocabularyConfiguration(sources=[training_dataset]))

    pipeline.train(
        output=str(tmp_path / "ner_experiment"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_dataset,
    )


def test_offset_from_tags():
    # TODO: Can be removed when we test the decode method with a custom TaskOutput above!
    nlp = spacy.load("en_core_web_sm")

    doc = nlp("Test this sheight")
    entities = offsets_from_tags(doc, tags=["O", "B-AIGHT", "L-AIGHT"])
    assert entities[0].keys() == dict(start=None, end=None, label=None, start_token=None, end_token=None).keys()
    assert entities[0]["label"] == "AIGHT"
    assert entities[0]["start"] == 5
    assert entities[0]["end"] == 17

    entities = offsets_from_tags(doc, tags=["O", "B-AIGHT", "L-AIGHT"], only_token_spans=True)
    assert entities[0].keys() == dict(label=None, start_token=None, end_token=None).keys()
    assert entities[0]["start_token"] == 1
    assert entities[0]["end_token"] == 3


def test_preserve_pretokenization(
    pipeline_dict, training_dataset, trainer_dict, tmp_path
):
    pipeline = Pipeline.from_config(pipeline_dict)
    tokens = ["test", "this", "pre tokenized", "text"]
    prediction = pipeline.predict(tokens)
    assert len(prediction["tags"][0]) == len(tokens)
