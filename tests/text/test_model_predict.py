import pytest

from biome.text import Pipeline
from biome.text._model import PipelineModel
from biome.text.configuration import PredictionConfiguration
from biome.text.errors import WrongInputError
from biome.text.featurizer import FeaturizeError
from biome.text.modules.heads.task_prediction import TaskPrediction


@pytest.fixture
def model() -> PipelineModel:
    pipeline = Pipeline.from_config(
        {
            "name": "test_predict",
            "head": {"type": "TextClassification", "labels": ["a"]},
        }
    )
    return pipeline._model


def test_training_mode_warning(model):
    with pytest.warns(UserWarning):
        model.predict([{"text": "test"}], PredictionConfiguration)
    assert model.training is False


def test_forward_pass_error(model, monkeypatch, caplog):
    def mock_text_to_instance(**kwargs):
        return "mock instance"

    def mock_forward_on_instances(*args, **kwargs):
        raise Exception("mock Exception")

    monkeypatch.setattr(model, "text_to_instance", mock_text_to_instance)
    monkeypatch.setattr(model, "forward_on_instances", mock_forward_on_instances)

    predictions = model.predict(
        [{"text": "Some value that breaks the forward pass"}], PredictionConfiguration
    )

    assert predictions == [None]
    assert len(caplog.record_tuples) == 2
    assert caplog.record_tuples[0] == ("biome.text._model", 40, "mock Exception")
    assert caplog.record_tuples[1] == (
        "biome.text._model",
        30,
        "Failed to make a forward pass for '[{'text': 'Some value that breaks the forward pass'}]'",
    )


def test_return_type(model, monkeypatch):
    def mock_make_task_prediction(*args, **kwargs):
        return TaskPrediction()

    monkeypatch.setattr(model.head, "make_task_prediction", mock_make_task_prediction)

    predictions = model.predict(
        [{"text": "test"}, {"text": "test2"}], PredictionConfiguration()
    )
    assert isinstance(predictions, list)
    assert all([isinstance(pred, TaskPrediction) for pred in predictions])


def test_text_to_instance(model, caplog):
    with pytest.raises(TypeError):
        model.text_to_instance(wrong_kwarg="wrong argument")

    with pytest.raises(TypeError):
        model.text_to_instance(label="missing required argument")

    model.text_to_instance(text="")
    assert caplog.record_tuples[0] == (
        "biome.text._model",
        30,
        "The provided input data contains empty strings/tokens: ",
    )
