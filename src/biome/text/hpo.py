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

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.errors import ValidationError

_LOGGER = logging.getLogger(__name__)


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
        By default (None), all metrics from Pytorch Lightning will be reported to Tune with the same name.
    vocab_config
        A `VocabularyConfiguration` to create/extend the pipeline's vocabulary.
        If `"default"` (str), we will use the default configuration `VocabularyConfiguration()`.
        If None, we will leave the pipeline's vocabulary untouched. Default: `"default"`.
    name
        Used as project name for the WandB logger and for the experiment name in the MLFlow logger.
        By default we construct following string: 'HPO on %date (%time)'
    trainable
        A custom trainable function that takes as input the `TuneExperiment.config` dict.
    silence
        If True, silence the biome.text logger. Default: False.
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
        trainer_config: TrainerConfiguration,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        vocab_config: Optional[Union[str, VocabularyConfiguration]] = "default",
        metrics: Union[None, str, List[str], Dict[str, str]] = None,
        name: Optional[str] = None,
        trainable: Optional[Callable] = None,
        silence: bool = False,
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
        self._trainer_config = asdict(trainer_config)
        vocab_config: Optional[VocabularyConfiguration] = (
            VocabularyConfiguration() if vocab_config == "default" else vocab_config
        )
        self._vocab_config: Optional[Dict] = (
            asdict(vocab_config) if vocab_config else vocab_config
        )

        self.trainable = trainable or self._default_trainable

        self._silence = silence

        self._name = name or f"HPO on {datetime.now().strftime('%Y-%m-%d (%I-%M)')}"
        if not os.environ.get("WANDB_PROJECT"):
            os.environ["WANDB_PROJECT"] = self._name

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

    @property
    def config(self) -> dict:
        """The config dictionary used by the `TuneExperiment.trainable` function"""
        return {
            "pipeline_config": self._pipeline_config,
            "trainer_config": self._trainer_config,
            "train_dataset_path": self._train_dataset_path,
            "valid_dataset_path": self._valid_dataset_path,
            "vocab_config": self._vocab_config,
            "silence": self._silence,
            "name": self._name,
            "metrics": self._metrics,
        }

    @staticmethod
    def _default_trainable(config, checkpoint_dir=None):
        """A default trainable function used by `tune.run`

        It performs the most straight forward training loop with the provided `config`:
        - Create the pipeline (optionally with a provided vocab)
        - Set up a TuneMetrics logger that reports all metrics back to ray tune after each epoch
        - Execute the training
        """
        if config["silence"]:
            logging.getLogger("biome.text").setLevel(logging.ERROR)

        pipeline = Pipeline.from_config(config["pipeline_config"])

        trainer_config = TrainerConfiguration(**config["trainer_config"])

        vocab_config = config["vocab_config"]
        if vocab_config:
            vocab_config = VocabularyConfiguration(**vocab_config)

        callbacks = trainer_config.callbacks
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        if not any(
            [isinstance(callback, TuneReportCallback) for callback in callbacks]
        ):
            tune_callback = TuneReportCallback(metrics=config["metrics"])
            if trainer_config.callbacks is None:
                trainer_config.callbacks = tune_callback
            else:
                trainer_config.callbacks = callbacks + [tune_callback]

        train_ds = Dataset.load_from_disk(config["train_dataset_path"])
        valid_ds = Dataset.load_from_disk(config["valid_dataset_path"])
        train_instances = train_ds.to_instances(pipeline=pipeline, disable_tqdm=True)
        valid_instances = valid_ds.to_instances(pipeline=pipeline, disable_tqdm=True)

        trainer = Trainer(
            pipeline=pipeline,
            train_dataset=train_instances,
            valid_dataset=valid_instances,
            trainer_config=trainer_config,
            vocab_config=vocab_config,
        )
        trainer.fit()

    def __del__(self):
        """Cleans up the created tmp dirs for the datasets and vocab"""
        for tmp_dir in self._created_tmp_dirs:
            tmp_dir.cleanup()
