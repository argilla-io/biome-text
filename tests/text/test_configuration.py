import pytest
import yaml
from allennlp.data.fields import ListField, TextField
from spacy.tokens.token import Token

from biome.text import Pipeline
from biome.text.configuration import (
    CharFeatures,
    FeaturesConfiguration,
    PipelineConfiguration,
    TaskHeadConfiguration,
    TokenizerConfiguration,
    WordFeatures,
)
from biome.text.modules.heads import TextClassification
from biome.text.modules.configuration.allennlp_configuration import (
    Seq2SeqEncoderConfiguration,
)


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

    assert pl.trainable_parameter_names == pl_yaml.trainable_parameter_names
    assert pl.trainable_parameters == pl_yaml.trainable_parameters

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
