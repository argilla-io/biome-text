import pytest

from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration
from biome.text.features import TransformersFeatures
from biome.text.data import DataSource
from pathlib import Path


@pytest.fixture
def train_data_source() -> DataSource:
    source = (
        Path(__file__).parent.parent
        / "resources"
        / "data"
        / "emotions_with_transformers.txt"
    )
    training_ds = DataSource(
        source=str(source), format="csv", sep=";", names=["text", "label"]
    )

    return training_ds


@pytest.fixture
def pipeline_dict() -> dict:
    pipeline_dict = {
        "name": "emotions_with_transformers",
        "features": {"transformers": {"model_name": "distilroberta-base"},},
        "head": {
            "type": "TextClassification",
            "labels": ["anger", "fear", "joy", "love", "sadness", "surprise",],
            "pooler": {
                "type": "bert_pooler",
                "pretrained_model": "distilroberta-base",
                "requires_grad": True,
                "dropout": 0.1,
            },
        },
    }

    return pipeline_dict


@pytest.fixture
def trainer_dict() -> dict:
    return {
        "batch_size": 16,
        "num_epochs": 1,
        "optimizer": {"type": "adam", "lr": 0.0001,},
    }


def test_pure_transformers(tmp_path, pipeline_dict, trainer_dict, train_data_source):
    pl = Pipeline.from_config(pipeline_dict)
    trainer = TrainerConfiguration(**trainer_dict)

    assert pl.backbone.vocab.get_vocab_size("transformers") == 50265

    pl.predict(text="test")

    output = tmp_path / "output"
    training_results = pl.train(
        output=str(output), trainer=trainer, training=train_data_source,
    )

    # test vocab from a pretrained file
    pl = Pipeline.from_pretrained(str(output / "model.tar.gz"))

    assert pl.backbone.vocab.get_vocab_size("transformers") == 50265


def test_transformers_and_word(tmp_path, pipeline_dict, trainer_dict, train_data_source):
    del pipeline_dict["head"]["pooler"]
    pipeline_dict["features"].update({"word": {"embedding_dim": 16, "lowercase_tokens": True}})

    pl = Pipeline.from_config(pipeline_dict)
    trainer = TrainerConfiguration(**trainer_dict)
    vocab = VocabularyConfiguration(sources=[train_data_source])
    pl.create_vocabulary(vocab)

    assert pl.backbone.vocab.get_vocab_size("transformers") == 50265
    assert pl.backbone.vocab.get_vocab_size("word") == 273

    pl.predict(text="test")

    output = tmp_path / "output"
    training_results = pl.train(
        output=str(output), trainer=trainer, training=train_data_source,
    )

    # test vocab from a pretrained file
    pl = Pipeline.from_pretrained(str(output / "model.tar.gz"))

    assert pl.backbone.vocab.get_vocab_size("transformers") == 50265
    assert pl.backbone.vocab.get_vocab_size("word") == 273


def test_max_length_not_affecting_shorter_sequences(pipeline_dict):
    pl = Pipeline.from_config(pipeline_dict)
    state_dict = pl._model.state_dict()
    probs = pl.predict("Test this")["probs"]

    pipeline_dict["features"]["transformers"]["max_length"] = 100
    pl = Pipeline.from_config(pipeline_dict)
    pl._model.load_state_dict(state_dict)
    probs_max_length = pl.predict("Test this")["probs"]

    assert all([log1 == log2 for log1, log2 in zip(probs, probs_max_length)])


def test_serialization(pipeline_dict):
    feature = TransformersFeatures(**pipeline_dict["features"]["transformers"])

    assert feature == TransformersFeatures(**feature.to_json())


