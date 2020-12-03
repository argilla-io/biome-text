from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration
from biome.text.features import TransformersFeatures


@pytest.fixture
def train_dataset() -> Dataset:
    """Creates the training dataset"""
    source = (
        Path(__file__).parent.parent
        / "resources"
        / "data"
        / "emotions_with_transformers.txt"
    )

    train_dataset = Dataset.from_csv(
        paths=str(source), delimiter=";", column_names=["text", "label"]
    )
    return train_dataset


@pytest.fixture
def pipeline_dict() -> dict:
    """Creation of pipeline dictionary"""

    pipeline_dict = {
        "name": "emotions_with_transformers",
        "features": {
            "transformers": {"model_name": "sshleifer/tiny-distilbert-base-cased"}
        },
        "head": {
            "type": "TextClassification",
            "labels": [
                "anger",
                "fear",
                "joy",
                "love",
                "sadness",
                "surprise",
            ],
            "pooler": {
                "type": "bert_pooler",
                "pretrained_model": "sshleifer/tiny-distilbert-base-cased",
                "requires_grad": True,
                "dropout": 0.1,
            },
        },
    }

    return pipeline_dict


@pytest.fixture
def trainer_dict() -> dict:
    """Creation of trainer dictionary"""

    return {
        "batch_size": 16,
        "num_epochs": 1,
        "optimizer": {
            "type": "adam",
            "lr": 0.0001,
        },
    }


def test_pure_transformers(tmp_path, pipeline_dict, trainer_dict, train_dataset):
    """Testing a Transformer training process and a model load"""

    pl = Pipeline.from_config(pipeline_dict)
    trainer = TrainerConfiguration(**trainer_dict)

    # Check a fixed vocabulary size for the model
    assert pl.backbone.vocab.get_vocab_size("transformers") == 28996

    pl.predict(text="test")

    output = tmp_path / "output"
    pl.train(output=str(output), trainer=trainer, training=train_dataset)

    # Test vocabulary from a pretrained file
    pl = Pipeline.from_pretrained(str(output / "model.tar.gz"))

    # Check a fixed vocabulary size for the model after loading
    assert pl.backbone.vocab.get_vocab_size("transformers") == 28996


def test_transformers_and_word(tmp_path, pipeline_dict, trainer_dict, train_dataset):
    """Testing Transformer pipeline with an added word feature layer"""
    # Changing the pipeline to delete the BERT pooler and add a word feature
    del pipeline_dict["head"]["pooler"]
    pipeline_dict["features"].update(
        {"word": {"embedding_dim": 16, "lowercase_tokens": True}}
    )

    pl = Pipeline.from_config(pipeline_dict)
    pl.predict(text="test")

    output = tmp_path / "output"
    trainer = TrainerConfiguration(**trainer_dict)
    pl.train(output=str(output), trainer=trainer, training=train_dataset)

    # Check a fixed vocabulary size for the transformer and the word feature
    assert pl.backbone.vocab.get_vocab_size("transformers") == 28996
    assert pl.backbone.vocab.get_vocab_size("word") == 273

    # Test vocab from a pretrained file
    pl = Pipeline.from_pretrained(str(output / "model.tar.gz"))

    # Check a fixed vocabulary size for the transformer and the word feature after loading
    assert pl.backbone.vocab.get_vocab_size("transformers") == 28996
    assert pl.backbone.vocab.get_vocab_size("word") == 273


def test_max_length_not_affecting_shorter_sequences(pipeline_dict):
    """Max length change should not affect at all previous shorter-length models"""

    pl = Pipeline.from_config(pipeline_dict)
    state_dict = pl._model.state_dict()  # dict with the whole state of the module
    probs = pl.predict("Test this")["probabilities"]  # probabilities of the test input

    pipeline_dict["features"]["transformers"]["max_length"] = 100  # changing max length
    pl = Pipeline.from_config(pipeline_dict)
    pl._model.load_state_dict(state_dict)  # loading previous state from dict
    probs_max_length = pl.predict("Test this")["probabilities"]

    assert_allclose(probs, probs_max_length)


def test_serialization(pipeline_dict):
    """Testing object saving. Model from the pipeline must be equal to the model from .json"""

    feature = TransformersFeatures(**pipeline_dict["features"]["transformers"])
    assert feature == TransformersFeatures(**feature.to_json())
