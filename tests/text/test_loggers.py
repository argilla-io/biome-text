import os
import sys
from tempfile import mkdtemp
from urllib.parse import urlparse

import mlflow
import pytest
from mlflow.utils import mlflow_tags

from biome.text import AllenNLPTrainerConfiguration
from biome.text import Pipeline
from biome.text import PipelineConfiguration
from biome.text.loggers import MlflowLogger
from biome.text.loggers import WandBLogger
from biome.text.modules.heads import TaskHeadConfiguration
from biome.text.modules.heads import TextClassification
from biome.text.training_results import TrainingResults


@pytest.fixture
def pipeline() -> Pipeline:
    return Pipeline.from_config(
        PipelineConfiguration(
            name="test-pipeline",
            head=TaskHeadConfiguration(type=TextClassification, labels=["A", "B"]),
        )
    )


def test_mlflow_logger(pipeline):

    logger = MlflowLogger(
        experiment_name="test-experiment", run_name="test_run", tag1="my-tag"
    )

    trainer = AllenNLPTrainerConfiguration()

    logger.init_train(pipeline, trainer, training=None)
    for epoch in range(0, 10):
        logger.log_epoch_metrics(epoch, metrics={"key": 10 * epoch})

    model_path = mkdtemp()
    metrics = {"metric": 200}
    logger.end_train(TrainingResults(model_path, metrics))

    run = mlflow.get_run(logger._run_id)
    assert run
    # Tags
    assert "test_run" == run.data.tags[mlflow_tags.MLFLOW_RUN_NAME]
    assert "my-tag" == run.data.tags["tag1"]
    # Parameters
    expected_parmams = {
        "pipeline.features.word.trainable": "True",
        "pipeline.num_parameters": "202",
        "pipeline.num_trainable_parameters": "202",
        "pipeline.features.word.embedding_dim": "50",
        "pipeline.head.type": "biome.text.modules.heads.classification.text_classification.TextClassification",
        "pipeline.head.labels": "['A', 'B']",
        "pipeline.name": "test-pipeline",
        "pipeline.tokenizer.lang": "en",
        "trainer.batch_size": "16",
        "trainer.validation_metric": "-loss",
        "trainer.optimizer.type": "adam",
        "trainer.patience": "2",
        "trainer.num_epochs": "20",
        "trainer.num_serialized_models_to_keep": "1",
        "pipeline.tokenizer.remove_space_tokens": "True",
    }
    assert expected_parmams == run.data.params
    # Artifacts
    assert os.path.basename(model_path) in os.listdir(
        urlparse(run.info.artifact_uri).path
    )
    # Metrics
    for metric in metrics:
        assert (
            metric in run.data.metrics and run.data.metrics[metric] == metrics[metric]
        )


@pytest.mark.skipif("wandb" not in sys.modules, reason="wandb client not installed")
class TestWandBLogger:
    def test_wandb_logger_init_warning(self, monkeypatch, capsys):
        import wandb

        monkeypatch.setattr(wandb.sdk.internal.internal_api.Api, "api_url", None)

        wandb.ensure_configured()
        WandBLogger(run_name="test_run", tags=["test_tag"])
        assert capsys.readouterr().err.startswith("wandb")

    def test_wandb_logger_init_train(
        self, monkeypatch, pipeline, change_to_tmp_working_dir
    ):
        # It's impossible to test wandb ... but these tests will soon go away when we have our lightning trainer ready
        class MockRun:
            def __init__(self, name, tags, dir, project, *args, **kwargs):
                self.name = name
                self.tags = tags
                self.dir = dir
                self.project = project

        def mock_wandb_init(*args, **kwargs):
            return MockRun(*args, **kwargs)

        import wandb

        monkeypatch.setattr(wandb, "init", mock_wandb_init)
        monkeypatch.setenv("WANDB_PROJECT", "mock_project")

        wandb.ensure_configured()
        logger = WandBLogger(run_name="test_run", tags=["test_tag"])

        logger.init_train(
            pipeline=pipeline,
            trainer_configuration=AllenNLPTrainerConfiguration(),
            training=None,
        )

        assert logger._run.name == "test_run"
        assert logger._run.tags[0] == "test_tag"
        assert logger._run.dir.startswith(".wandb")
        assert logger._run.project == "mock_project"
