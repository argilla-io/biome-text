import pytest

from biome.text import Pipeline
from biome.text._model import PipelineModel
from biome.text.configuration import PredictionConfiguration
from biome.text.errors import WrongInputError
from biome.text.errors import WrongValueError
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


def test_value_error(model, monkeypatch):
    def mock_text_to_instance(**kwargs):
        return None

    monkeypatch.setattr(model, "text_to_instance", mock_text_to_instance)

    with pytest.raises(WrongValueError):
        model.predict(
            [{"text": "Some value that breaks the featurize"}], PredictionConfiguration
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


def test_text_to_instance(model):
    with pytest.raises(WrongInputError):
        model.text_to_instance(wrong_kwarg="wrong argument")
    with pytest.raises(WrongInputError):
        model.text_to_instance(label="missing required argument")
