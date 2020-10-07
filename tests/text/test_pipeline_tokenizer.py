from biome.text import Pipeline
from biome.text.configuration import TokenizerConfiguration
from biome.text.tokenizer import Tokenizer, TransformersTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.common.checks import ConfigurationError
import pytest


@pytest.fixture
def pipeline_dict(request) -> dict:
    """Pipeline config dict. You need to update the labels!"""
    pipeline_dict = {
        "name": "transformers_tokenizer_test",
        "features": {
            "transformers": {
                "model_name": "sshleifer/tiny-distilroberta-base"
            }
        },
        "head": {
            "type": "TextClassification",
            "labels": ["a", "b"],
        },
    }
    return pipeline_dict


def test_pipeline_transformers_tokenizer(pipeline_dict):
    pl = Pipeline.from_config(pipeline_dict)

    assert pl.config.tokenizer_config.transformers_kwargs == {"model_name": "sshleifer/tiny-distilroberta-base"}
    assert not pl.config.features.transformers.is_mismatched
    assert type(pl.backbone.featurizer.indexer["transformers"]) is PretrainedTransformerIndexer
    assert type(pl.backbone.tokenizer) is TransformersTokenizer

    prediction = pl.predict("Test this!")


def test_pipeline_default_tokenizer(pipeline_dict):
    pipeline_dict["features"].update({"word": {"embedding_dim": 2}})
    pl = Pipeline.from_config(pipeline_dict)

    assert pl.config.tokenizer_config == TokenizerConfiguration()
    assert pl.config.features.transformers.is_mismatched
    assert type(pl.backbone.featurizer.indexer["transformers"]) is PretrainedTransformerMismatchedIndexer
    assert type(pl.backbone.tokenizer) is Tokenizer

    prediction = pl.predict("Test this!")


def test_invalid_tokenizer_features_combination(pipeline_dict):
    pipeline_dict["features"].update({"word": {"embedding_dim": 2}})
    pipeline_dict["tokenizer"] = {"transformers_kwargs": {"model_name": "sshleifer/tiny-distilroberta-base"}}

    with pytest.raises(ConfigurationError):
        pl = Pipeline.from_config(pipeline_dict)
