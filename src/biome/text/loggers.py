import dataclasses
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from allennlp.training import EpochCallback
from allennlp.training import GradientDescentTrainer
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags

from biome.text.dataset import InstancesDataset
from biome.text.training_results import TrainingResults

# We do not require wandb
try:
    import wandb
except ImportError:
    _HAS_WANDB = False
else:
    wandb.ensure_configured()
    _HAS_WANDB = True


class BaseTrainLogger(EpochCallback):
    """Base train logger for pipeline training"""

    def init_train(
        self,
        pipeline: "Pipeline",
        trainer_configuration: "TrainerConfiguration",
        training: InstancesDataset,
        validation: Optional[InstancesDataset] = None,
        test: Optional[InstancesDataset] = None,
    ):
        """Init train logging

        Parameters
        ----------
        pipeline
            The training pipeline
        trainer_configuration
            The trainer configuration
        training
            Training dataset
        validation
            Validation dataset
        test
            Test dataset
        """
        pass

    def end_train(self, results: TrainingResults):
        """End train logging

        Parameters
        ----------
        results
            The training result set
        """
        pass

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch metrics

        Parameters
        ----------
        epoch
            The current epoch
        metrics
            The metrics related to current epoch
        """
        pass

    def __call__(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ):
        if epoch >= 0:
            self.log_epoch_metrics(epoch, metrics)


class MlflowLogger(BaseTrainLogger):
    """A common mlflow logger for pipeline training

    Parameters
    ----------
    experiment_name
        The experiment name
    artifact_location
        The artifact location used for this experiment
    run_name
        If specified, set a name to created run
    tags
        Extra arguments used as tags to created experiment run
    """

    __LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        experiment_name: str = None,
        artifact_location: str = None,
        run_name: str = None,
        **tags,
    ):
        self._client = MlflowClient()
        self._experiment = self._configure_experiment_with_retry(
            experiment_name, artifact_location
        )

        tags = tags or {}
        if run_name:
            tags[mlflow_tags.MLFLOW_RUN_NAME] = run_name

        run = self._client.create_run(self._experiment.experiment_id, tags=tags)
        self._run_id = run.info.run_id

        self._skipped_metrics = ["training_duration"]

    def _configure_experiment_with_retry(
        self, experiment_name: str, artifact_location: str, retries: int = 5
    ) -> Optional[Experiment]:
        """Tries to configure (fetch or create) an mlflow experiment with retrying process on errors"""
        if retries <= 0:
            return None
        try:
            experiment = self._client.get_experiment_by_name(
                experiment_name or "default"
            )
            if experiment:
                return experiment

            return self._client.get_experiment(
                self._client.create_experiment(experiment_name, artifact_location)
            )
        except Exception as e:
            self.__LOGGER.debug(e)
            return self._configure_experiment_with_retry(
                experiment_name, artifact_location, retries=retries - 1
            )

    def init_train(
        self,
        pipeline: "Pipeline",
        trainer_configuration: "TrainerConfiguration",
        training: InstancesDataset,
        validation: Optional[InstancesDataset] = None,
        test: Optional[InstancesDataset] = None,
    ):
        from pandas import json_normalize

        for prefix, params_set in [
            ("pipeline", json_normalize(pipeline.config.as_dict())),
            ("trainer", json_normalize(dataclasses.asdict(trainer_configuration))),
        ]:
            for key, value in params_set.to_dict(orient="records")[0].items():
                if value:
                    self._client.log_param(self._run_id, f"{prefix}.{key}", value)
        self._client.log_param(
            self._run_id, key="pipeline.num_parameters", value=pipeline.num_parameters
        )
        self._client.log_param(
            self._run_id,
            key="pipeline.num_trainable_parameters",
            value=pipeline.num_trainable_parameters,
        )

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):

        [
            self._client.log_metric(self._run_id, key=k, value=v, step=epoch)
            for k, v in metrics.items()
            if k not in self._skipped_metrics
        ]

    def end_train(self, results: TrainingResults):
        try:
            self._client.log_artifact(self._run_id, local_path=results.model_path)
            [
                self._client.log_metric(self._run_id, key=k, value=v)
                for k, v in results.metrics.items()
                if k not in self._skipped_metrics
            ]
        finally:
            self._client.set_terminated(self._run_id)


class WandBLogger(BaseTrainLogger):
    """Logger for WandB

    Parameters
    ----------
    project_name
        Name of your WandB project
    run_name
        Name of your run
    tags
        Extra arguments used as tags to created experiment run
    """

    def __init__(
        self, project_name: str = "biome", run_name: str = None, tags: List[str] = None
    ):
        if wandb.api.api_key is None:
            wandb.termwarn(
                "W&B installed but not logged in. "
                "Run `wandb login` or `import wandb; wandb.login()` or set the WANDB_API_KEY env variable."
            )

        self.project_name = project_name
        self.run_name = run_name
        self.tags = tags

        self._run = None

    def init_train(
        self,
        pipeline: "Pipeline",
        trainer_configuration: "TrainerConfiguration",
        training: InstancesDataset,
        validation: Optional[InstancesDataset] = None,
        test: Optional[InstancesDataset] = None,
    ):
        config = {
            "pipeline": pipeline.config.as_dict(),
            "trainer": dataclasses.asdict(trainer_configuration),
        }
        config["pipeline"]["num_parameters"] = pipeline.num_parameters
        config["pipeline"][
            "num_trainable_parameters"
        ] = pipeline.num_trainable_parameters
        self._run = wandb.init(
            project=self.project_name, name=self.run_name, tags=self.tags, config=config
        )

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        wandb.log(metrics)

    def end_train(self, results: TrainingResults):
        if self._run:
            self._run.finish()


def is_wandb_installed_and_logged_in() -> bool:
    """Checks if wandb is installed and if a login is detected.

    Returns
    -------
    bool
        Is true, if wandb is installed and a login is detected, otherwise false.
    """
    if not _HAS_WANDB:
        return False
    if wandb.api.api_key is None:
        wandb.termwarn(
            "W&B installed but not logged in. "
            "Run `wandb login` or `import wandb; wandb.login()` or set the WANDB_API_KEY env variable."
        )
        return False
    return True


def add_default_wandb_logger_if_needed(
    loggers: List[BaseTrainLogger],
) -> List[BaseTrainLogger]:
    """Adds the default WandBLogger if a WandB login is detected and no WandBLogger is found in `loggers`.

    Parameters
    ----------
    loggers
        List of loggers used in the training

    Returns
    -------
    loggers
        List of loggers with a default WandBLogger at position 0 if needed
    """
    if any([isinstance(logger, WandBLogger) for logger in loggers]):
        pass
    elif not is_wandb_installed_and_logged_in():
        pass
    else:
        loggers = [WandBLogger()] + loggers

    return loggers
