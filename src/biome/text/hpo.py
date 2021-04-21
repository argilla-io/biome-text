"""
This module includes all components related to an HPO experiment execution.
It tries to allow for a simple integration with the HPO library 'Ray Tune'.
"""
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import mlflow
from allennlp.data import Vocabulary
from pytorch_lightning import Callback
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from biome.text import AllenNLPTrainerConfiguration
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration
from biome.text import helpers
from biome.text.dataset import Dataset
from biome.text.errors import ValidationError
from biome.text.loggers import BaseTrainLogger
from biome.text.loggers import MlflowLogger
from biome.text.loggers import WandBLogger
from biome.text.loggers import is_wandb_installed_and_logged_in

_LOGGER = logging.getLogger(__name__)


class TuneMetricsLogger(BaseTrainLogger):
    """
    A trainer logger defined for sending validation metrics to ray tune system. Normally, those
    metrics will be used by schedulers for trial experiments stop.
    """

    @staticmethod
    def _metric_should_be_reported(metric_name: str) -> bool:
        """Determines if a metric should be reported"""
        return (
            not metric_name.startswith("validation__")
            and metric_name.startswith("validation_")
        ) or (
            not metric_name.startswith("best_validation__")
            and metric_name.startswith("best_validation_")
        )

    def log_epoch_metrics(self, epoch, metrics):
        # fmt: off
        tune.report(**{
            k: v
            for k, v in metrics.items()
            if self._metric_should_be_reported(k)
        })
        # fmt: on


class TuneExperiment(tune.Experiment):
    """This class provides a trainable function and a config to conduct an HPO with `ray.tune.run`

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
    metrics
        Metrics to report to Tune. If this is a list, each item describes the metric key reported to PyTorch Lightning,
        and it will be reported under the same name to Tune. If this is a dict, each key will be the name reported to
        Tune and the respective value will be the metric key reported to PyTorch Lightning.
        By default (None), all metrics from Pytorch Lightnint will be reported to Tune with the same name.
    vocab
        If you want to share the same vocabulary between the trials you can provide it here
    name
        Used as project name for the WandB logger and for the experiment name in the MLFlow logger.
        By default we construct following string: 'HPO on %date (%time)'
    trainable
        A custom trainable function that takes as input the `TuneExperiment.config` dict.
    mlflow
        If True (default), logs HPO to MLFlow.
    wandb
        If True (default), logs HPO to WandB.
    **kwargs
        The rest of the kwargs are passed on to `tune.Experiment.__init__`.
        They must not contain the 'name', 'run' or the 'config' key,
        since these are provided automatically by `TuneExperiment`.

    Attributes
    ----------
    trainable
        The trainable function used by ray tune
    config
        The config dict passed on to the trainable function

    Examples
    --------
    A minimal usage would be:

    >>> from biome.text import Dataset, TrainerConfiguration
    >>> from ray import tune
    >>> pipeline_config = {
    ...     "name": "tune_experiment_example",
    ...     "head": {"type": "TextClassification", "labels": ["a", "b"]},
    ... }
    >>> trainer_config = TrainerConfiguration(
    ...     optimizer={"type": "adam", "lr": tune.loguniform(1e-3, 1e-2)},
    ...     progress_bar_refresh_rate=0
    ... )
    >>> train_dataset = Dataset.from_dict({"text": ["test", "this"], "label": ["a", "b"]})
    >>> valid_dataset = Dataset.from_dict({"text": ["test", "this"], "label": ["a", "b"]})
    >>> my_exp = TuneExperiment(pipeline_config, trainer_config, train_dataset, valid_dataset, num_samples=10)
    >>> tune.run(my_exp) # doctest: +SKIP

    """

    def __init__(
        self,
        pipeline_config: dict,
        trainer_config: Union[dict, TrainerConfiguration],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        metrics: Union[None, str, List[str], Dict[str, str]] = None,
        vocab: Optional[Vocabulary] = None,
        name: Optional[str] = None,
        trainable: Optional[Callable] = None,
        mlflow: bool = True,
        wandb: bool = True,
        **kwargs,
    ):
        if (
            "name" in kwargs.keys()
            or "run" in kwargs.keys()
            or "config" in kwargs.keys()
        ):
            raise ValueError(
                f"Your `kwargs` must not contain the 'name', 'run' or 'config' key."
                f"These are provided automatically by `TuneExperiment`."
            )

        # save created tmp dirs in this list to clean them up when object gets destroyed
        self._created_tmp_dirs: List[tempfile.TemporaryDirectory] = []

        self._train_dataset_path = self._save_dataset_to_disk(train_dataset)
        self._valid_dataset_path = self._save_dataset_to_disk(valid_dataset)

        self._pipeline_config = pipeline_config
        if isinstance(trainer_config, dict):
            _LOGGER.warning(
                "Training with the AllenNLP trainer will be removed in the next version, "
                "please use a `biome.text.TrainerConfiguration` in the `trainer_config`."
            )
            self._trainer_config = trainer_config or {}
            self.trainable = trainable or self._allennlp_trainable
        else:
            self._trainer_config = asdict(trainer_config)
            self.trainable = trainable or self._default_trainable

        self._vocab_path = (
            self._save_vocab_to_disk(vocab) if vocab is not None else None
        )
        self._name = name or f"HPO on {datetime.now().strftime('%Y-%m-%d (%I-%M)')}"
        if not os.environ.get("WANDB_PROJECT"):
            os.environ["WANDB_PROJECT"] = self._name

        self._mlflow = mlflow
        self._wandb = wandb
        self._metrics = metrics

        super().__init__(
            name=self._name, run=self.trainable, config=self.config, **kwargs
        )

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
        """The config dictionary used by the `TuneExperiment.trainable` function"""
        return {
            "pipeline_config": self._pipeline_config,
            "trainer_config": self._trainer_config,
            "train_dataset_path": self._train_dataset_path,
            "valid_dataset_path": self._valid_dataset_path,
            "mlflow": self._mlflow,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "wandb": self._wandb,
            "vocab_path": self._vocab_path,
            "name": self._name,
            "metrics": self._metrics,
        }

    @staticmethod
    def _allennlp_trainable(config, reporter, checkpoint_dir=None):
        """A default trainable function used by `tune.run`

        It performs the most straight forward training loop with the provided `config`:
        - Create the pipeline (optionally with a provided vocab)
        - Set up a MLFlow and WandB logger
        - Set up a TuneMetrics logger that reports all metrics back to ray tune after each epoch
        - Execute the training
        """
        pipeline = Pipeline.from_config(
            config["pipeline_config"], vocab_path=config["vocab_path"]
        )

        trainer_config = AllenNLPTrainerConfiguration(
            **helpers.sanitize_for_params(config["trainer_config"])
        )

        train_ds = Dataset.load_from_disk(config["train_dataset_path"])
        valid_ds = Dataset.load_from_disk(config["valid_dataset_path"])

        train_loggers = [TuneMetricsLogger()]
        if config["mlflow"]:
            mlflow_tracking_uri = config["mlflow_tracking_uri"]
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            train_loggers = [
                MlflowLogger(
                    experiment_name=config["name"],
                    run_name=reporter.trial_name,
                    ray_trial_id=reporter.trial_id,
                    ray_logdir=reporter.logdir,
                )
            ] + train_loggers
        if config["wandb"] and is_wandb_installed_and_logged_in():
            train_loggers = [WandBLogger(project_name=config["name"])] + train_loggers

        pipeline.train(
            output="output",
            training=train_ds,
            validation=valid_ds,
            trainer=trainer_config,
            loggers=train_loggers,
            vocab_config=None if config["vocab_path"] else "default",
        )

    @staticmethod
    def _default_trainable(config, checkpoint_dir=None):
        """A default trainable function used by `tune.run`

        It performs the most straight forward training loop with the provided `config`:
        - Create the pipeline (optionally with a provided vocab)
        - Set up a TuneMetrics logger that reports all metrics back to ray tune after each epoch
        - Execute the training
        """
        pipeline = Pipeline.from_config(
            config["pipeline_config"], vocab_path=config["vocab_path"]
        )

        trainer_config = TrainerConfiguration(**config["trainer_config"])

        tune_callback = TuneReportCallback(metrics=config["metrics"])
        if trainer_config.callbacks is None:
            trainer_config.callbacks = [tune_callback]
        if isinstance(trainer_config.callbacks, Callback):
            trainer_config.callbacks = [trainer_config.callbacks, tune_callback]
        elif isinstance(trainer_config.callbacks, list):
            trainer_config.callbacks.append(tune_callback)

        train_ds = Dataset.load_from_disk(config["train_dataset_path"])
        valid_ds = Dataset.load_from_disk(config["valid_dataset_path"])
        train_instances = train_ds.to_instances(pipeline=pipeline, disable_tqdm=True)
        valid_instances = valid_ds.to_instances(pipeline=pipeline, disable_tqdm=True)

        trainer = Trainer(
            pipeline=pipeline,
            train_dataset=train_instances,
            valid_dataset=valid_instances,
            trainer_config=trainer_config,
            vocab_config=None if config["vocab_path"] else "default",
        )
        trainer.fit()

    def __del__(self):
        """Cleans up the created tmp dirs for the datasets and vocab"""
        for tmp_dir in self._created_tmp_dirs:
            tmp_dir.cleanup()
