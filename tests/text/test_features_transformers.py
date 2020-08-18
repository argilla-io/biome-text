import pytest

from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration
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


def test_train(tmp_path, pipeline_dict, trainer_dict, train_data_source):
    pl = Pipeline.from_config(pipeline_dict)
    trainer = TrainerConfiguration(**trainer_dict)
    vocab = VocabularyConfiguration(sources=[train_data_source])
    pl.create_vocabulary(vocab)

    assert pl.backbone.vocab.get_vocab_size("transformers") == 50265

    pl.predict(text="test")

    output = tmp_path / "output"

    training_results = pl.train(
        output=str(output), trainer=trainer, training=train_data_source,
    )

    # test vocab from a pretrained file
    pl = Pipeline.from_pretrained(str(output / "model.tar.gz"))

    assert pl.backbone.vocab.get_vocab_size("transformers") == 50265
