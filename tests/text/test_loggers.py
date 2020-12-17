import os
from tempfile import mkdtemp
from urllib.parse import urlparse

import mlflow
from mlflow.utils import mlflow_tags

from biome.text import Pipeline
from biome.text import PipelineConfiguration
from biome.text import TrainerConfiguration
from biome.text.loggers import MlflowLogger
from biome.text.modules.heads import TaskHeadConfiguration
from biome.text.modules.heads import TextClassification
from biome.text.training_results import TrainingResults


def test_mlflow_logger():

    logger = MlflowLogger(
        experiment_name="test-experiment", run_name="test_run", tag1="my-tag"
    )

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
