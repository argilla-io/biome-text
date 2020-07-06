from typing import Any, Dict, Optional

from allennlp.training import EpochCallback, GradientDescentTrainer
from mlflow.tracking import MlflowClient

from biome.text.data import InstancesDataset
from biome.text.training_results import TrainingResults


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
        """
        Init train logging

        Parameters
        ----------
        pipeline:
            The training pipeline
        trainer_configuration:
            The trainer configuration
        training:
            Training dataset
        validation:
            Validation dataset
        test:
            Test dataset

        """
        pass

    def end_train(self, results: TrainingResults):
        """
        End train logging

        Parameters
        ----------
        results:
            The training result set

        """
        pass

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log epoch metrics

        Parameters
        ----------

        epoch:
            The current epoch
        metrics:
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
    """A common mlflow logger for pipeline training"""

    def __init__(self, experiment_name: str = None, artifact_location: str = None):
        self._client = MlflowClient()
        self._experiment = self._client.get_experiment_by_name(
            experiment_name or "default"
        )
        if not self._experiment:
            self._experiment = self._client.get_experiment(
                self._client.create_experiment(experiment_name, artifact_location)
            )

        self._run_id = self._client.create_run(
            self._experiment.experiment_id
        ).info.run_id
        self._skipped_metrics = ["training_duration"]

    def init_train(
        self,
        pipeline: "Pipeline",
        trainer_configuration: "TrainerConfiguration",
        training: InstancesDataset,
        validation: Optional[InstancesDataset] = None,
        test: Optional[InstancesDataset] = None,
    ):
        # fmt: off
        self._client.log_param(self._run_id, key="name", value=pipeline.name)
        self._client.log_param(self._run_id, key="pipeline", value=pipeline.config.as_dict())
        self._client.log_param(self._run_id, key="num_parameters", value=pipeline.trainable_parameters)
        self._client.log_param(self._run_id, key="trainer", value=trainer_configuration)
        # fmt: on

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
