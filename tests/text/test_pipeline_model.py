from typing import Any
from typing import Dict

import numpy
import pytest
from allennlp.data import Instance

from biome.text import Pipeline
from biome.text import PipelineConfiguration
from biome.text.backbone import ModelBackbone
from biome.text.configuration import FeaturesConfiguration
from biome.text.modules.heads import TaskHeadConfiguration
from biome.text.modules.heads import TextClassification


class TestHead(TextClassification):
    def __init__(self, backbone: ModelBackbone):
        super(TestHead, self).__init__(backbone, labels=["test", "notest"])

    def explain_prediction(
        self, prediction: Dict[str, numpy.array], instance: Instance, n_steps: int
    ) -> Dict[str, Any]:
        return {}


class TestHeadWithRaise(TestHead):
    def explain_prediction(
        self, prediction: Dict[str, numpy.array], instance: Instance, n_steps: int
    ) -> Dict[str, Any]:
        raise NotImplementedError


def test_explain_tokenized_as_default():
    pipeline_config = PipelineConfiguration(
        name="test-classifier",
        head=TaskHeadConfiguration(type=TestHead),
        features=FeaturesConfiguration(),
    )
    pipeline = Pipeline.from_config(pipeline_config)
    prediction = pipeline.explain("This is a simple test with only tokens in explain")
    explain = prediction["explain"]

    assert explain
    assert explain.get("text")
    for token_info in explain["text"]:
        assert isinstance(token_info.get("token"), str)
        assert token_info.get("attribution") == 0.0


def test_explain_without_steps():
    pipeline_config = PipelineConfiguration(
        name="test-classifier",
        head=TaskHeadConfiguration(type=TestHeadWithRaise),
        features=FeaturesConfiguration(),
    )
    pipeline = Pipeline.from_config(pipeline_config)
    with pytest.raises(NotImplementedError):
        pipeline.explain("This is a simple test with only tokens in explain")

    prediction = pipeline.explain(
        "This is a simple test with only tokens in explain", n_steps=0
    )
    assert "explain" in prediction


def test_predict_batch():
    pipeline_config = PipelineConfiguration(
        name="test-classifier",
        head=TaskHeadConfiguration(type=TestHead),
        features=FeaturesConfiguration(),
    )
    pipeline = Pipeline.from_config(pipeline_config)
    predictions = pipeline.predict_batch([{"text": "test1"}, {"text": "test2"}])

    assert len(predictions) == 2
    assert all([isinstance(prediction, dict) for prediction in predictions])


def test_explain_batch():
    pipeline_config = PipelineConfiguration(
        name="test-classifier",
        head=TaskHeadConfiguration(type=TestHead),
        features=FeaturesConfiguration(),
    )
    pipeline = Pipeline.from_config(pipeline_config)
    predictions = pipeline.explain_batch([{"text": "test1"}, {"text": "test2"}])
    assert len(predictions) == 2

    for prediction in predictions:
        explain: Dict[str, Any] = prediction["explain"]

        assert explain
        assert explain.get("text")
        for token_info in explain["text"]:
            assert isinstance(token_info.get("token"), str)
            assert token_info.get("attribution") == 0.0
