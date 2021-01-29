import pytest

from biome.text._helpers import PipelineTrainer
from biome.text.training_results import TrainingResults


@pytest.fixture
def deactivate_pipeline_trainer(monkeypatch):
    def mock_init(*args, **kwargs):
        pass

    monkeypatch.setattr(PipelineTrainer, "__init__", mock_init)

    def mock_train(*args, **kwargs):
        return TrainingResults(
            model_path="mock_output_path", metrics={"mock_metric": 0.0}
        )

    monkeypatch.setattr(PipelineTrainer, "train", mock_train)
