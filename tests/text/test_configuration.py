import pytest
import yaml
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields import ListField
from allennlp.data.fields import TextField
from spacy.tokens.token import Token

from biome.text import Pipeline
from biome.text.configuration import CharFeatures
from biome.text.configuration import FeaturesConfiguration
from biome.text.configuration import PipelineConfiguration
from biome.text.configuration import TaskHeadConfiguration
from biome.text.configuration import TokenizerConfiguration
from biome.text.configuration import WordFeatures
from biome.text.modules.configuration.allennlp_configuration import (
    Seq2SeqEncoderConfiguration,
)
from biome.text.modules.heads import TextClassification


@pytest.fixture
def pipeline_yaml(tmp_path):
    pipeline_dict = {
        "name": "test_pipeline_config",
        "tokenizer": {
            "text_cleaning": {"rules": ["strip_spaces"]},
            "use_spacy_tokens": True,
        },
        "features": {
            "word": {"embedding_dim": 2, "lowercase_tokens": True},
            "char": {
                "embedding_dim": 2,
                "encoder": {
                    "type": "gru",
                    "hidden_size": 2,
                    "num_layers": 1,
                    "bidirectional": True,
                },
                "dropout": 0.1,
            },
        },
        "encoder": {
            "type": "gru",
            "hidden_size": 2,
            "num_layers": 1,
            "bidirectional": True,
        },
        "head": {
            "type": "TextClassification",
            "labels": ["duplicate", "not_duplicate"],
            "pooler": {"type": "boe"},
        },
    }

    yaml_path = tmp_path / "pipeline.yml"
    with yaml_path.open("w") as file:
        yaml.safe_dump(pipeline_dict, file)

    return str(yaml_path)


@pytest.fixture
def transformers_pipeline_config():
    return {
        "name": "transformers_tokenizer_plus_tokenclassification",
        "features": {
            "transformers": {"model_name": "sshleifer/tiny-distilroberta-base"}
        },
        "head": {
            "type": "TextClassification",
            "labels": ["duplicate", "not_duplicate"],
        },
    }


def test_pipeline_without_word_features():
    tokenizer_config = TokenizerConfiguration()
    char_features = CharFeatures(
        embedding_dim=2,
        encoder={
            "type": "gru",
            "hidden_size": 2,
            "num_layers": 1,
            "bidirectional": True,
        },
        dropout=0.1,
    )
    features_config = FeaturesConfiguration(char=char_features)
    encoder_spec = Seq2SeqEncoderConfiguration(
        type="gru", hidden_size=2, num_layers=1, bidirectional=True
    )

    head_spec = TaskHeadConfiguration(
        type="TextClassification",
        labels=["duplicate", "not_duplicate"],
        pooler={"type": "boe"},
    )

    pipeline_config = PipelineConfiguration(
        name="no_word_features",
        head=head_spec,
        features=features_config,
        tokenizer=tokenizer_config,
        encoder=encoder_spec,
    )

    pl = Pipeline.from_config(pipeline_config)
    assert "word" not in pl.backbone.featurizer.indexer
    assert "char" in pl.backbone.featurizer.indexer


def test_pipeline_config(pipeline_yaml):
    tokenizer_config = TokenizerConfiguration(
        text_cleaning={"rules": ["strip_spaces"]}, use_spacy_tokens=True
    )

    word_features = WordFeatures(embedding_dim=2, lowercase_tokens=True)
    char_features = CharFeatures(
        embedding_dim=2,
        encoder={
            "type": "gru",
            "hidden_size": 2,
            "num_layers": 1,
            "bidirectional": True,
        },
        dropout=0.1,
    )
    features_config = FeaturesConfiguration(word=word_features, char=char_features)

    encoder_spec = Seq2SeqEncoderConfiguration(
        type="gru", hidden_size=2, num_layers=1, bidirectional=True
    )

    head_spec = TaskHeadConfiguration(
        type=TextClassification,
        labels=["duplicate", "not_duplicate"],
        pooler={"type": "boe"},
    )

    pipeline_config = PipelineConfiguration(
        name="test_pipeline_config",
        head=head_spec,
        features=features_config,
        tokenizer=tokenizer_config,
        encoder=encoder_spec,
    )

    pl = Pipeline.from_config(pipeline_config)

    pl_yaml = Pipeline.from_yaml(pipeline_yaml)

    assert pl.named_trainable_parameters == pl_yaml.named_trainable_parameters
    assert pl.num_trainable_parameters == pl_yaml.num_trainable_parameters
    assert pl.num_parameters == pl_yaml.num_parameters

    sample_text = "My simple text"
    for instance in [
        pl.backbone.featurizer(sample_text),
        pl_yaml.backbone.featurizer(sample_text),
    ]:
        for key, value in instance.items():
            assert key == "record"
            assert isinstance(value, ListField)
            assert len(value) == 1
            for text in value:
                assert isinstance(text, TextField)
                assert all(map(lambda t: isinstance(t, Token), text.tokens))
                assert sample_text == " ".join([t.text for t in text.tokens])


def test_invalid_tokenizer_features_combination(transformers_pipeline_config):
    transformers_pipeline_config["features"].update({"word": {"embedding_dim": 2}})
    transformers_pipeline_config["tokenizer"] = {"use_transformers": True}

    with pytest.raises(ConfigurationError):
        Pipeline.from_config(transformers_pipeline_config)


def test_not_implemented_transformers_with_tokenclassification(
    transformers_pipeline_config,
):
    transformers_pipeline_config["tokenizer"] = {"use_transformers": True}
    transformers_pipeline_config["head"] = {
        "type": "TokenClassification",
        "labels": ["NER"],
    }
    with pytest.raises(NotImplementedError):
        Pipeline.from_config(transformers_pipeline_config)


def test_invalid_transformers_tokenizer_indexer_embedder_combination(
    transformers_pipeline_config,
):
    transformers_pipeline_config["tokenizer"] = {
        "transformers_kwargs": {"model_name": "distilroberta-base"}
    }

    with pytest.raises(ConfigurationError):
        Pipeline.from_config(transformers_pipeline_config)
