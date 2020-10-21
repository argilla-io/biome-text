"""
This module includes all components related to a HPO experiment execution.
It tries to allow for a simple integration with HPO libraries like Ray Tune.
"""
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Union, List, Optional

import mlflow
from allennlp.data import Vocabulary
from ray import tune

from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration, helpers
from biome.text.data import DataSource
from biome.text.dataset import Dataset
from biome.text.errors import ValidationError
from biome.text.loggers import (
    BaseTrainLogger,
    MlflowLogger,
    WandBLogger,
    is_wandb_installed_and_logged_in,
)


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
        MlflowLogger(
            experiment_name=experiment_name,
            run_name=reporter.trial_name,
            ray_trial_id=reporter.trial_id,
            ray_logdir=reporter.logdir,
        ),
        TuneMetricsLogger(),
    ]
    if is_wandb_installed_and_logged_in():
        train_loggers = [WandBLogger(project_name=experiment_name)] + train_loggers

    shared_vocab.save_to_files("vocabulary") if shared_vocab else None
    pipeline = Pipeline.from_config(pipeline_config, vocab_path="vocabulary")
    trainer_config = TrainerConfiguration(**helpers.sanitize_for_params(trainer_config))

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
    name
        The experiment name used for experiment logging organization
    pipeline
        `Pipeline` used as base pipeline for hpo
    train
        The train data source location
    validation
        The validation data source location
    trainer
        `TrainerConfiguration` used as base trainer config for hpo
    hpo_params
        `HpoParams` selected for hyperparameter sampling.
    shared_vocab
        If true, pipeline vocab will be used for all trials in this experiment.
        Otherwise, the vocab will be generated using input data sources in each trial.
        This could be desired if some hpo defined param affects to vocab creation.
        Defaults: False
    trainable_fn
        Function defining the hpo training flow. Normally the default function should
        be enough for common use cases. Anyway, you can provide your own trainable function.
        In this case, it's your responsibility to report tune metrics for a successful hpo
        Defaults: `tune_hpo_train`
    num_samples
        Number of times to sample from the hyperparameter space.
        If `grid_search` is provided as an argument in the hpo_params, the grid will be repeated `num_samples` of times.
        Default: 1
    """

    name: str
    pipeline: Pipeline
    train: str
    validation: str
    trainer: TrainerConfiguration = field(default_factory=TrainerConfiguration)
    hpo_params: HpoParams = field(default_factory=HpoParams)
    shared_vocab: bool = False
    trainable_fn: Callable = field(default_factory=lambda: tune_hpo_train)
    num_samples: int = 1

    def as_tune_experiment(self) -> tune.Experiment:
        config = {
            "name": self.name,
            "train": self.train,
            "validation": self.validation,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "pipeline": helpers.merge_dicts(
                self.hpo_params.pipeline,
                helpers.sanitize_for_params(self.pipeline.config.as_dict())
            ),
            "trainer": helpers.merge_dicts(
                self.hpo_params.trainer,
                helpers.sanitize_for_params(asdict(self.trainer))
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


class RayTuneTrainable:
    """This class provides a trainable function and a config to conduct an HPO with `ray.tune.run`

    Minimal usage:
    >>> my_trainable = RayTuneTrainable(pipeline_config, trainer_config, train_dataset, valid_dataset)
    >>> tune.run(my_trainable.func, config=my_trainable.config)

    Parameters
    ----------
    pipeline_config
        The pipeline configuration with its hyperparemter search spaces:
        https://docs.ray.io/en/master/tune/key-concepts.html#search-spaces
    trainer_config
        The trainer configuration with its hyperparameter search spaces
    train_dataset
        Training dataset
    valid_dataset
        Validation dataset
    vocab
        If you want to share the same vocabulary between the trials you can provide it here
    name
        Used as the project name in the WandB logger and as experiment name in the MLFlow logger.
        By default we construct following string: 'HPO on %date (%time)'
    """

    def __init__(
        self,
        pipeline_config: dict,
        trainer_config: dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        vocab: Optional[Vocabulary] = None,
        name: Optional[str] = None
    ):
        # save created tmp dirs in this list to clean them up when object gets destroyed
        self._created_tmp_dirs: List[tempfile.TemporaryDirectory] = []

        self._train_dataset_path = self._save_dataset_to_disk(train_dataset)
        self._valid_dataset_path = self._save_dataset_to_disk(valid_dataset)

        self._pipeline_config = pipeline_config
        self._trainer_config = trainer_config or {}
        self._vocab_path = (
            self._save_vocab_to_disk(vocab) if vocab is not None else None
        )
        self._name = name or f"HPO on {datetime.now().strftime('%Y-%m-%d (%I-%M)')}"

    def _save_dataset_to_disk(self, dataset: Dataset) -> str:
        """Saves the dataset to disk if not saved already

        Parameters
        ----------
        dataset
            Dataset to save to disk

        Returns
        -------
        dataset_path
            Path to the saved dataset, that is a directory
        """
        try:
            filename = Path(dataset.dataset.cache_files[0]["filename"])
        except (IndexError, KeyError):
            filename = Path()

        if filename.name != "dataset.arrow":
            tmp_dir = tempfile.TemporaryDirectory()
            self._created_tmp_dirs.append(tmp_dir)
            dataset_path = tmp_dir.name
            dataset.save_to_disk(dataset_path)
        else:
            dataset_path = str(filename.absolute())

        # Make sure that we can load the dataset successfully
        try:
            Dataset.load_from_disk(dataset_path)
        except Exception as exception:
            raise ValidationError(
                f"Could not load dataset saved in '{dataset_path}'"
            ) from exception

        return dataset_path

    def _save_vocab_to_disk(self, vocab: Vocabulary) -> str:
        """Saves the vocab to disk to reuse it between the trials

        Parameters
        ----------
        vocab
            Vocabulary to be saved to disk

        Returns
        -------
        vocab_path
            Path to the vocabulary, that is a directory
        """
        tmp_dir = tempfile.TemporaryDirectory()
        self._created_tmp_dirs.append(tmp_dir)
        vocab_path = tmp_dir.name
        vocab.save_to_files(vocab_path)

        # Make sure that we can load the vocab successfully
        try:
            Vocabulary.from_files(vocab_path)
        except Exception as exception:
            raise ValidationError(
                f"Could not load vocab saved in '{vocab_path}'"
            ) from exception

        return vocab_path

    @property
    def config(self) -> dict:
        """The config dictionary used by the `RayTuneTrainable.func` function"""
        return {
            "pipeline_config": self._pipeline_config,
            "trainer_config": self._trainer_config,
            "train_dataset_path": self._train_dataset_path,
            "valid_dataset_path": self._valid_dataset_path,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "vocab_path": self._vocab_path,
            "name": self._name,
        }

    @staticmethod
    def func(config, reporter):
        """The trainable function passed on to `ray.tune.run`

        It performs the most straight forward training loop with the provided `config`:
        - Create the pipeline (optionally with a provided vocab)
        - Set up a MLFlow and WandB logger
        - Set up a TuneMetrics logger that reports all metrics back to ray tune after each epoch
        - Create the vocab if necessary
        - Execute the training
        """
        pipeline = Pipeline.from_config(
            config["pipeline_config"], vocab_path=config["vocab_path"]
        )

        trainer_config = TrainerConfiguration(
            **helpers.sanitize_for_params(config["trainer_config"])
        )

        mlflow_tracking_uri = config["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        train_ds = Dataset.load_from_disk(config["train_dataset_path"])
        valid_ds = Dataset.load_from_disk(config["valid_dataset_path"])

        train_loggers = [
            MlflowLogger(
                experiment_name=config["name"],
                run_name=reporter.trial_name,
                ray_trial_id=reporter.trial_id,
                ray_logdir=reporter.logdir,
            ),
            TuneMetricsLogger(),
        ]
        if is_wandb_installed_and_logged_in():
            train_loggers = [WandBLogger(project_name=config["name"])] + train_loggers

        if pipeline.has_empty_vocab():
            vocab_config = VocabularyConfiguration(sources=[train_ds, valid_ds])
            pipeline.create_vocabulary(vocab_config)

        pipeline.train(
            output="training",
            training=train_ds,
            validation=valid_ds,
            trainer=trainer_config,
            loggers=train_loggers,
        )

    def __del__(self):
        """Cleans up the created tmp dirs with the datasets"""
        for tmp_dir in self._created_tmp_dirs:
            tmp_dir.cleanup()
