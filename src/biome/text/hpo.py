"""
This module includes all components related to a HPO experiment execution.
It tries to allow for a simple integration with HPO libraries like Ray Tune.
"""
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional

import mlflow
from allennlp.data import Vocabulary
from ray import tune

from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import helpers
from biome.text.dataset import Dataset
from biome.text.errors import ValidationError
from biome.text.loggers import BaseTrainLogger
from biome.text.loggers import MlflowLogger
from biome.text.loggers import WandBLogger
from biome.text.loggers import is_wandb_installed_and_logged_in


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

    Minimal usage:
    >>> my_exp = TuneExperiment(pipeline_config, trainer_config, train_dataset, valid_dataset)
    >>> tune.run(my_exp)

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
        Used for the `tune.Experiment.name`, the project name in the WandB logger
        and for the experiment name in the MLFlow logger.
        By default we construct following string: 'HPO on %date (%time)'
    trainable
        A custom trainable function that takes as input the `TuneExperiment.config` dict.
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
    """

    def __init__(
        self,
        pipeline_config: dict,
        trainer_config: dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        vocab: Optional[Vocabulary] = None,
        name: Optional[str] = None,
        trainable: Optional[Callable] = None,
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
        self._trainer_config = trainer_config or {}
        self._vocab_path = (
            self._save_vocab_to_disk(vocab) if vocab is not None else None
        )
        self._name = name or f"HPO on {datetime.now().strftime('%Y-%m-%d (%I-%M)')}"

        self.trainable = trainable or self._default_trainable

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
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "vocab_path": self._vocab_path,
            "name": self._name,
        }

    @staticmethod
    def _default_trainable(config, reporter):
        """A default trainable function used by `tune.run`

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

        pipeline.train(
            output="training",
            training=train_ds,
            validation=valid_ds,
            trainer=trainer_config,
            loggers=train_loggers,
            vocab_config=None if config["vocab_path"] else "default",
        )

    def __del__(self):
        """Cleans up the created tmp dirs for the datasets and vocab"""
        for tmp_dir in self._created_tmp_dirs:
            tmp_dir.cleanup()
