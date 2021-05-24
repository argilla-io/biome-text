import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import VocabularyConfiguration
from biome.text.errors import EmptyVocabError
from biome.text.features import CharFeatures
from biome.text.features import TransformersFeatures
from biome.text.features import WordFeatures


@pytest.fixture
def pipeline():
    config = {
        "name": "vocab_test",
        "features": {
            "transformers": {"model_name": "sshleifer/tiny-distilbert-base-cased"},
            "word": {"embedding_dim": 2},
            "char": {
                "embedding_dim": 2,
                "dropout": 0.1,
                "encoder": {
                    "type": "gru",
                    "hidden_size": 2,
                    "num_layers": 1,
                    "bidirectional": False,
                },
            },
        },
        "head": {
            "type": "TextClassification",
            "labels": ["good", "bad"],
        },
    }

    return Pipeline.from_config(config)


@pytest.fixture
def train_dataset():
    data = {"text": ["this is a test", "and another one"], "label": ["good", "bad"]}
    return Dataset.from_dict(data)


@pytest.fixture
def valid_dataset():
    data = {
        "text": ["and what about the validation", "do not forget this one"],
        "label": ["bad", "good"],
    }
    return Dataset.from_dict(data)


def test_default_vocab(pipeline, train_dataset, valid_dataset):
    # Transformer vocab is added on pipeline creation
    assert pipeline.vocab.get_vocab_size(TransformersFeatures.namespace) == 28996
    # While word and char vocab should be empty (except for the oov and padding token)
    assert pipeline.vocab.get_vocab_size(WordFeatures.namespace) == 2
    assert pipeline.vocab.get_vocab_size(CharFeatures.namespace) == 2

    # Training should build a default vocab with only the training dataset
    Trainer(pipeline, train_dataset=train_dataset)
    assert pipeline.vocab.get_vocab_size(WordFeatures.namespace) == 9
    assert pipeline.vocab.get_vocab_size(CharFeatures.namespace) == 12
    assert pipeline.vocab.get_vocab_size(TransformersFeatures.namespace) == 28996

    # Pretrained pipelines should extend the vocab by default
    Trainer(pipeline, train_dataset=valid_dataset)
    assert pipeline.vocab.get_vocab_size(WordFeatures.namespace) == 16
    assert pipeline.vocab.get_vocab_size(CharFeatures.namespace) == 19
    assert pipeline.vocab.get_vocab_size(TransformersFeatures.namespace) == 28996


def test_specific_vocab_config(pipeline, train_dataset, valid_dataset):
    vocab_config = VocabularyConfiguration(include_valid_data=True)

    Trainer(
        pipeline,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        vocab_config=vocab_config,
    )
    assert pipeline.vocab.get_vocab_size(WordFeatures.namespace) == 16
    assert pipeline.vocab.get_vocab_size(CharFeatures.namespace) == 19
    assert pipeline.vocab.get_vocab_size(TransformersFeatures.namespace) == 28996


def test_not_touching_vocab(pipeline, train_dataset, valid_dataset):
    # vocab_config=None leaves the pipeline's vocab empty from an unpretrained pipeline
    with pytest.raises(EmptyVocabError):
        Trainer(pipeline, train_dataset=train_dataset, vocab_config=None)

    # vocab_config=None should not extend the vocab for a pretrained pipeline
    Trainer(pipeline, train_dataset=train_dataset)
    assert pipeline.vocab.get_vocab_size(WordFeatures.namespace) == 9
    assert pipeline.vocab.get_vocab_size(CharFeatures.namespace) == 12
    Trainer(pipeline, train_dataset=valid_dataset, vocab_config=None)
    assert pipeline.vocab.get_vocab_size(WordFeatures.namespace) == 9
    assert pipeline.vocab.get_vocab_size(CharFeatures.namespace) == 12
