import pytest

from biome.text import Pipeline
from biome.text.modules.heads.task_prediction import TextClassificationPrediction


@pytest.fixture
def pipeline() -> Pipeline:
    return Pipeline.from_config(
        {
            "name": "test_predict",
            "head": {"type": "TextClassification", "labels": ["a"]},
        }
    )


def test_raise_value_error(pipeline):
    with pytest.raises(ValueError):
        pipeline.predict()
    with pytest.raises(ValueError):
        pipeline.predict("test", batch=[{"text": "test"}])
    with pytest.raises(ValueError):
        pipeline.predict(text="test", batch=[{"text": "test"}])


def test_map_args_kwargs_to_input():
    class MockPipeline:
        def __init__(self, inputs):
            self._inputs = inputs

        @property
        def inputs(self):
            return self._inputs

    assert Pipeline._map_args_kwargs_to_input(MockPipeline(["text"]), "test") == {
        "text": "test"
    }
    assert Pipeline._map_args_kwargs_to_input(MockPipeline(["text"]), text="test") == {
        "text": "test"
    }
    assert Pipeline._map_args_kwargs_to_input(
        MockPipeline(["text", "text2"]), "test", text2="test2"
    ) == {"text": "test", "text2": "test2"}


def test_return_single_or_list(pipeline, monkeypatch):
    def mock_predict(batch, prediction_config):
        return [
            TextClassificationPrediction(labels=["a"], probabilities=[1])
            if i % 2 == 0
            else None
            for i, _ in enumerate(batch)
        ]

    monkeypatch.setattr(pipeline._model, "predict", mock_predict)

    assert isinstance(pipeline.predict("test"), dict)

    batch_prediction = pipeline.predict(batch=[{"text": "test"}])
    assert isinstance(batch_prediction, list) and len(batch_prediction) == 1
    assert isinstance(batch_prediction[0], dict)

    batch_prediction = pipeline.predict(
        batch=[{"text": "test"}, {"text": "no instance for this input"}]
    )
    assert isinstance(batch_prediction, list) and len(batch_prediction) == 2
    assert isinstance(batch_prediction[0], dict) and batch_prediction[1] is None
