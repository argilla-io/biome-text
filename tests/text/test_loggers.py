import os
from tempfile import mkdtemp
from urllib.parse import urlparse

import mlflow
from mlflow.utils import mlflow_tags

from biome.text import Pipeline, PipelineConfiguration, TrainerConfiguration
from biome.text.loggers import MlflowLogger
from biome.text.modules.heads import TaskHeadConfiguration, TextClassification
from biome.text.training_results import TrainingResults


def test_mlflow_logger():

    logger = MlflowLogger(experiment_name="test-experiment", run_name="test_run", tag1="my-tag")

    pipeline = Pipeline.from_config(
        PipelineConfiguration(
            name="test-pipeline",
            head=TaskHeadConfiguration(type=TextClassification, labels=["A", "B"]),
        )
    )
    trainer = TrainerConfiguration()

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
    assert pipeline.name == run.data.params["name"]
    assert str(pipeline.trainable_parameters) == run.data.params["num_parameters"]
    assert str(trainer) == run.data.params["trainer"]
    assert str(pipeline.config.as_dict()) == run.data.params["pipeline"]
    # Artifacts
    assert os.path.basename(model_path) in os.listdir(
        urlparse(run.info.artifact_uri).path
    )
    # Metrics
    for metric in metrics:
        assert (
            metric in run.data.metrics and run.data.metrics[metric] == metrics[metric]
        )
