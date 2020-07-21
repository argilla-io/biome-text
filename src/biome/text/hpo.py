"""
This module includes all components related to hpo experiment execution. Tries be allows a
simple integration with hyper-parameter optimization modules like Ray Tune
"""
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict

import mlflow
from allennlp.common.util import sanitize
from ray import tune

from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration, helpers
from biome.text.data import DataSource
from biome.text.loggers import BaseTrainLogger, MlflowLogger


class TuneMetricsLogger(BaseTrainLogger):
    """
    A trainer logger defined for sending validation metrics to ray tune system. Normally, those
    metrics will be used by schedulers for trial experiments stop.
    """

    @staticmethod
    def _metric_should_be_reported(metric_name: str) -> bool:
        """Determines if a metric should be reported"""
        # fmt:off
        return (
            not metric_name.startswith("validation__")
            and metric_name.startswith("validation_")
        )
        # fmt: on

    def log_epoch_metrics(self, epoch, metrics):
        # fmt: off
        tune.report(**{
            k: v
            for k, v in metrics.items()
            if self._metric_should_be_reported(k)
        })
        # fmt: on


def tune_hpo_train(config, reporter):
    """
    The main trainable method. This method defines common flow for hpo training.

    See `HpoExperiment` for details about input parameters
    """
    experiment_name = config["name"]
    mlflow_tracking_uri = config["mlflow_tracking_uri"]

    pipeline_config = config["pipeline"]
    trainer_config = config["trainer"]
    shared_vocab = config.get("vocab")

    train_source = config["train"]
    validation_source = config["validation"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    train_loggers = [
        TuneMetricsLogger(),
        MlflowLogger(
            experiment_name=experiment_name,
            run_name=reporter.trial_name,
            ray_trial_id=reporter.trial_id,
            ray_logdir=reporter.logdir,
        ),
    ]

    shared_vocab.save_to_files("vocabulary") if shared_vocab else None
    pipeline = Pipeline.from_config(pipeline_config, vocab_path="vocabulary")
    trainer_config = TrainerConfiguration(**trainer_config)

    train_ds = pipeline.create_dataset(DataSource(train_source))
    valid_ds = pipeline.create_dataset(DataSource(validation_source))

    if pipeline.has_empty_vocab():
        vocab_config = VocabularyConfiguration(sources=[train_ds, valid_ds])
        pipeline.create_vocabulary(vocab_config)

    pipeline.train(
        output="training",
        training=train_ds,
        validation=valid_ds,
        trainer=trainer_config,
        loggers=train_loggers,
        quiet=True,
    )


@dataclass
class HpoParams:
    """
    This class defines pipeline and trainer parameters selected for
    hyperparameter optimization sampling.

    Attributes
    ----------
    pipeline:
        A selection of pipeline parameters used for tune sampling
    trainer:
        A selection of trainer parameters used for tune sampling
    """

    pipeline: Dict[str, Any] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    # vocab: Dict[str, Any] = None


@dataclass
class HpoExperiment:
    """
    The hyper parameter optimization experiment data class

    Attributes
    ----------

    name:
        The experiment name used for experiment logging organization
    pipeline:
        `Pipeline` used as base pipeline for hpo
    train:
        The train data source location
    validation:
        The validation data source location
    trainer:
        `TrainerConfiguration` used as base trainer config for hpo
    hpo_params:
        `HpoParams` selected for hyperparameter sampling.
    shared_vocab:
        If true, pipeline vocab will be used for all trials in this experiment.
        Otherwise, the vocab will be generated using input data sources in each trial.
        This could be desired if some hpo defined param affects to vocab creation.
        Defaults: True
    trainable_fn:
        Function defining the hpo training flow. Normally the default function should
        be enough for common use cases. Anyway, you can provide your own trainable function.
        In this case, it's your responsibility to report tune metrics for a successful hpo
        Defaults: `tune_hpo_train`
    num_samples:
        This param addresses directly `ray.tune.run` `num_samples` argument, and sets
        the number of times to sample from the hyperparameter space. Default: 5

    """

    name: str
    pipeline: Pipeline
    train: str
    validation: str
    trainer: TrainerConfiguration = field(default_factory=TrainerConfiguration)
    hpo_params: HpoParams = field(default_factory=HpoParams)
    shared_vocab: bool = True
    trainable_fn: Callable = field(default_factory=lambda: tune_hpo_train)
    num_samples: int = 5

    def as_tune_experiment(self) -> tune.Experiment:
        config = {
            "name": self.name,
            "train": self.train,
            "validation": self.validation,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "pipeline": helpers.merge_dicts(
                self.hpo_params.pipeline,
                helpers.sanitize_for_params(self.pipeline.config.as_dict()),
            ),
            "trainer": helpers.merge_dicts(
                self.hpo_params.trainer,
                asdict(self.trainer),
            ),
        }
        if self.shared_vocab:
            config["vocab"] = self.pipeline.backbone.vocab

        return tune.Experiment(
            name=self.name,
            run=self.trainable_fn,
            config=config,
            local_dir=os.path.abspath("runs/tune"),
            num_samples=self.num_samples,
        )
