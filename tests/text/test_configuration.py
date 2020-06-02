import pytest
import yaml

from biome.text import Pipeline
from biome.text.configuration import (
    CharFeatures,
    FeaturesConfiguration,
    PipelineConfiguration,
    TaskHeadSpec,
    TokenizerConfiguration,
    WordFeatures,
)
from biome.text.modules.heads import TextClassification
from biome.text.modules.specs.allennlp_specs import Seq2SeqEncoderSpec


@pytest.fixture
def pipeline_yaml(tmp_path):
    pipeline_dict = {
        "name": "test_pipeline_config",
        "tokenizer": {"text_cleaning": {"rules": ["strip_spaces"]},},
        "features": {
            "word": {"embedding_dim": 2, "lowercase_tokens": True,},
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
            "pooler": {"type": "boe",},
        },
    }

    yaml_path = tmp_path / "pipeline.yml"
    with yaml_path.open("w") as file:
        yaml.safe_dump(pipeline_dict, file)

    return str(yaml_path)


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
    encoder_spec = Seq2SeqEncoderSpec(
        type="gru", hidden_size=2, num_layers=1, bidirectional=True
    )

    head_spec = TaskHeadSpec(
        type="TextClassification",
        labels=["duplicate", "not_duplicate"],
        pooler={"type": "boe"},
    )

    pipeline_config = PipelineConfiguration(
        name="no_word_features",
        tokenizer=tokenizer_config,
        features=features_config,
        encoder=encoder_spec,
        head=head_spec,
    )

    pl = Pipeline.from_config(pipeline_config)
    assert "word" not in pl.backbone.featurizer.indexer
    assert "char" in pl.backbone.featurizer.indexer


def test_pipeline_config(pipeline_yaml):
    tokenizer_config = TokenizerConfiguration(text_cleaning={"rules": ["strip_spaces"]})

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

    encoder_spec = Seq2SeqEncoderSpec(
        type="gru", hidden_size=2, num_layers=1, bidirectional=True
    )

    head_spec = TaskHeadSpec(
        type=TextClassification,
        labels=["duplicate", "not_duplicate"],
        pooler={"type": "boe"},
    )

    pipeline_config = PipelineConfiguration(
        name="test_pipeline_config",
        tokenizer=tokenizer_config,
        features=features_config,
        encoder=encoder_spec,
        head=head_spec,
    )

    pl = Pipeline.from_config(pipeline_config)

    pl_yaml = Pipeline.from_yaml(pipeline_yaml)

    assert pl.trainable_parameter_names == pl_yaml.trainable_parameter_names
    assert pl.trainable_parameters == pl_yaml.trainable_parameters

    sample_text = "My simple text"
    assert pl.backbone.featurizer(sample_text) == pl_yaml.backbone.featurizer(
        sample_text
    )
