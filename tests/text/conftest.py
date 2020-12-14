import pytest

from biome.text._helpers import PipelineTrainer


@pytest.fixture
def deactivate_pipeline_trainer(monkeypatch):
    def mock_init(*args, **kwargs):
        pass

    monkeypatch.setattr(PipelineTrainer, "__init__", mock_init)

    def mock_train(*args, **kwargs):
        return "mock_output_path", {"mock_metric": 0.0}

    monkeypatch.setattr(PipelineTrainer, "train", mock_train)
