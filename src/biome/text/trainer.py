import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import PyTorchDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.training.optimizers import Optimizer
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torch.utils.data import IterableDataset
from transformers.optimization import get_constant_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup

from biome.text.configuration import TrainerConfiguration
from biome.text.configuration import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.dataset import InstanceDataset

if TYPE_CHECKING:
    from biome.text.model import PipelineModel
    from biome.text.pipeline import Pipeline

# We do not require wandb
_HAS_WANDB = False
try:
    import wandb
except ImportError:
    pass
else:
    wandb.ensure_configured()
    _HAS_WANDB = True


_LOGGER = logging.getLogger(__name__)


class Trainer:
    """Class for training and testing a `biome.text.Pipeline`.

    It is basically a light wrapper around the awesome Pytorch Lightning Trainer to define custom defaults and
    facilitate the interaction with our pipelines.

    Parameters
    ----------
    pipeline
        Pipeline to train
    train_dataset
        The training dataset. Default: `None`.
    valid_dataset
        The validation dataset. Default: `None`.
    trainer_config
        The configuration of the trainer. Default: `TrainerConfiguration()`.
    vocab_config
        A `VocabularyConfiguration` to create/extend the pipeline's vocabulary.
        If `"default"` (str), we will use the default configuration `VocabularyConfiguration()`.
        If None, we will leave the pipeline's vocabulary untouched. Default: `"default"`.
    lazy
        If True, instances are lazily loaded from disk, otherwise they are loaded into memory.
        Ignored when passing in `InstanceDataset`s. Default: False.
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        train_dataset: Optional[Union[Dataset, InstanceDataset]] = None,
        valid_dataset: Optional[Union[Dataset, InstanceDataset]] = None,
        trainer_config: Optional[TrainerConfiguration] = None,
        vocab_config: Optional[Union[str, VocabularyConfiguration]] = "default",
        lazy: bool = False,
    ):
        self._pipeline = pipeline
        # since we will make changes to the config, better to make a copy -> asdict returns a deep copy
        self._trainer_config = (
            TrainerConfiguration(**asdict(trainer_config))
            if trainer_config is not None
            else TrainerConfiguration()
        )

        # Use GPU by default if available
        if self._trainer_config.gpus is None and torch.cuda.is_available():
            self._trainer_config.gpus = 1

        self._vocab_config: Optional[VocabularyConfiguration] = (
            VocabularyConfiguration() if vocab_config == "default" else vocab_config
        )
        self._lazy = lazy

        # create instances
        if isinstance(train_dataset, Dataset):
            self._train_instances = train_dataset.to_instances(
                self._pipeline, lazy=self._lazy, tqdm_desc="Loading training instances"
            )
        else:
            self._train_instances = train_dataset

        if isinstance(valid_dataset, Dataset):
            self._valid_instances = valid_dataset.to_instances(
                self._pipeline,
                lazy=self._lazy,
                tqdm_desc="Loading validation instances",
            )
        else:
            self._valid_instances = valid_dataset

        # Maybe we just want to call `self.test`
        if self._train_instances is not None:
            self._setup_for_training()

        self.trainer = pl.Trainer(**self._trainer_config.lightning_params)

    def _setup_for_training(self):
        """Create vocab, configure default loggers/callbacks, create optimizer/lr scheduler, setup best metrics"""
        # create vocab
        if self._vocab_config is not None:
            vocab_datasets = [self._train_instances]
            if (
                self._valid_instances is not None
                and self._vocab_config.include_valid_data
            ):
                vocab_datasets += [self._valid_instances]
            self._pipeline.create_vocab(vocab_datasets, config=self._vocab_config)

        # we give some special attention to these loggers/callbacks
        self._wandb_logger: Optional[WandbLogger] = None
        self._model_checkpoint: Optional[ModelCheckpoint] = None

        # add default callbacks/loggers
        self._trainer_config.callbacks = self._add_default_callbacks()
        if self._trainer_config.logger is not False:
            self._trainer_config.logger = self._add_default_loggers()

        # create optimizer, has to come AFTER creating the vocab!
        self._pipeline.model.optimizer = Optimizer.from_params(
            Params(
                {
                    "model_parameters": self._pipeline.model.named_parameters(),
                    **self._trainer_config.optimizer,
                }
            )
        )

        # create lr scheduler, has to come AFTER creating the optimizer!
        if not (
            self._trainer_config.warmup_steps == 0
            and self._trainer_config.lr_decay is None
        ):
            self._pipeline.model.lr_scheduler = self._create_lr_scheduler()
        else:
            self._pipeline.model.lr_scheduler = None

        # set monitor and mode for best validation metrics
        self._pipeline.model.monitor = self._trainer_config.monitor
        self._pipeline.model.monitor_mode = self._trainer_config.monitor_mode

    def _add_default_loggers(self) -> List[LightningLoggerBase]:
        """Adds optional default loggers and returns the extended list

        Added loggers: CSV, TensorBoard, WandB
        """
        loggers = self._trainer_config.logger
        if loggers is True:
            loggers = []
        elif isinstance(loggers, LightningLoggerBase):
            loggers = [loggers]

        def get_loggers_of_type(logger_type) -> List[LightningLoggerBase]:
            return [logger for logger in loggers if isinstance(logger, logger_type)]

        # csv
        if self._trainer_config.add_csv_logger and not get_loggers_of_type(CSVLogger):
            loggers.append(
                CSVLogger(
                    save_dir=self._trainer_config.default_root_dir or os.getcwd(),
                    name="csv",
                )
            )

        # tensorboard
        if self._trainer_config.add_tensorboard_logger and not get_loggers_of_type(
            TensorBoardLogger
        ):
            loggers.append(
                TensorBoardLogger(
                    save_dir=self._trainer_config.default_root_dir,
                    name="tensorboard",
                )
            )

        # wandb
        if (
            self._trainer_config.add_wandb_logger
            and _HAS_WANDB
            and not get_loggers_of_type(WandbLogger)
        ):
            self._wandb_logger = WandbLogger(
                save_dir=self._trainer_config.default_root_dir,
                project=os.environ.get("WANDB_PROJECT", "biome"),
            )
            loggers.append(self._wandb_logger)
        elif get_loggers_of_type(WandbLogger):
            self._wandb_logger = get_loggers_of_type(WandbLogger)[0]
        # somehow the wandb dir does not get created, i think this is a bug on pl side, have to check it out
        if self._wandb_logger is not None and not os.path.isdir(
            os.path.join(self._wandb_logger.save_dir, "wandb")
        ):
            os.makedirs(os.path.join(self._wandb_logger.save_dir, "wandb"))

        return loggers

    def _add_default_callbacks(self) -> List[Callback]:
        """Adds optional default callbacks and returns the extended list

        Added callbacks: ModelCheckpoint, EarlyStopping, LearningRateMonitor
        """
        callbacks = self._trainer_config.callbacks or []
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]

        def get_callbacks_of_type(callback_type) -> List[Callback]:
            return [
                callback
                for callback in callbacks
                if isinstance(callback, callback_type)
            ]

        # model checkpoint
        if self._trainer_config.checkpoint_callback and not get_callbacks_of_type(
            ModelCheckpoint
        ):
            monitor = self._trainer_config.monitor if self._valid_instances else None
            mode = self._trainer_config.monitor_mode
            save_top_k = (
                self._trainer_config.save_top_k_checkpoints
                if self._valid_instances
                else None
            )
            self._model_checkpoint = ModelCheckpointWithVocab(
                save_top_k=save_top_k, monitor=monitor, mode=mode
            )
            callbacks.append(self._model_checkpoint)
        elif get_callbacks_of_type(ModelCheckpoint):
            self._model_checkpoint = get_callbacks_of_type(ModelCheckpoint)[0]

        # early stopping
        if (
            self._trainer_config.add_early_stopping
            and self._valid_instances is not None
            and not get_callbacks_of_type(EarlyStopping)
        ):
            callbacks.append(
                EarlyStopping(
                    monitor=self._trainer_config.monitor,
                    mode=self._trainer_config.monitor_mode,
                    patience=self._trainer_config.patience,
                )
            )

        # lr monitor
        if self._trainer_config.add_lr_monitor is None and (
            self._trainer_config.warmup_steps != 0
            or self._trainer_config.lr_decay is not None
        ):
            self._trainer_config.add_lr_monitor = True
        if self._trainer_config.add_lr_monitor and not get_callbacks_of_type(
            LearningRateMonitor
        ):
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        return callbacks

    def _create_lr_scheduler(self) -> Dict:
        """Returns one of three default schedulers

        Possibilities: constant/linear/cosine schedule with or without warmup
        """
        steps_per_epoch = math.ceil(
            len(self._train_instances) / self._trainer_config.batch_size
        )
        try:
            training_steps = min(
                self._trainer_config.max_steps,
                self._trainer_config.max_epochs * steps_per_epoch,
            )
        # One or both of the max_* is None:
        except TypeError:
            training_steps = (
                self._trainer_config.max_steps
                or self._trainer_config.max_epochs * steps_per_epoch
                or 1000 * steps_per_epoch  # default of the lightning trainer
            )

        if self._trainer_config.lr_decay == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=self._pipeline.model.optimizer,
                num_warmup_steps=self._trainer_config.warmup_steps,
                num_training_steps=training_steps,
            )
        elif self._trainer_config.lr_decay == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=self._pipeline.model.optimizer,
                num_warmup_steps=self._trainer_config.warmup_steps,
                num_training_steps=training_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(
                optimizer=self._pipeline.model.optimizer,
                num_warmup_steps=self._trainer_config.warmup_steps,
            )

        return {
            "scheduler": scheduler,
            "interval": "step",
            "name": "learning_rate",
        }

    def fit(
        self, output_dir: Optional[Union[str, Path]] = "output", exist_ok: bool = False
    ):
        """Train the pipeline

        At the end of the training the pipeline will load the weights from the best checkpoint.

        Parameters
        ----------
        output_dir
            If specified, save the trained pipeline to this directory. Default: 'output'.
        exist_ok
            If True, overwrite the content of `output_dir`. Default: False.
        """
        if self._train_instances is None:
            _LOGGER.error(
                "You need training data to fit your model, please provide it on `self.__init__`."
            )
            return

        try:
            output_dir = Path(output_dir)
        except TypeError:
            output_dir = None
        else:
            output_dir.mkdir(exist_ok=exist_ok)

        # create dataloaders
        train_dataloader = create_dataloader(
            self._train_instances,
            batch_size=self._trainer_config.batch_size,
            data_bucketing=self._trainer_config.data_bucketing,
            num_workers=self._trainer_config.num_workers_for_dataloader,
        )
        valid_dataloader = (
            create_dataloader(
                self._valid_instances,
                batch_size=self._trainer_config.batch_size,
                data_bucketing=self._trainer_config.data_bucketing,
                num_workers=self._trainer_config.num_workers_for_dataloader,
            )
            if self._valid_instances is not None
            else None
        )

        # log config to wandb
        if self._wandb_logger is not None:
            config = {
                "pipeline": self._pipeline.config.as_dict(),
                "trainer": self._trainer_config.as_dict(),
            }
            self._wandb_logger.experiment.config.update(config)

        # training
        try:
            self.trainer.fit(
                self._pipeline.model,
                train_dataloader=train_dataloader,
                val_dataloaders=valid_dataloader,
            )
        finally:
            if self._model_checkpoint:
                self._load_best_weights()
            if output_dir:
                self._pipeline.save(output_dir)
                with (output_dir / "metrics.json").open("w") as file:
                    json.dump(sanitize(self.trainer.logged_metrics), file)
                self._trainer_config.to_yaml(output_dir / "trainer_config.yaml")
            if self._wandb_logger is not None:
                self._wandb_logger.experiment.finish()

    def _load_best_weights(self):
        """Load weights from the best model checkpoint"""
        checkpoint_path = self._model_checkpoint.best_model_path
        if checkpoint_path:
            _LOGGER.debug("Loading best weights")
            checkpoint = pl_load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
            self._pipeline.model.load_state_dict(checkpoint["state_dict"])

    def test(
        self,
        test_dataset: Union[Dataset, InstanceDataset],
        batch_size: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate your model on a test dataset

        Parameters
        ----------
        test_dataset
            The test data set.
        batch_size
            The batch size. If None (default), we will use the batch size specified in the `TrainerConfiguration`.
        output_dir
            Save a `metrics.json` to this output directory. Default: None.
        verbose
            If True, prints the test results. Default: True.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the metrics
        """
        # load instances
        if isinstance(test_dataset, Dataset):
            self._train_instances = test_dataset.to_instances(
                self._pipeline, lazy=self._lazy, tqdm_desc="Loading test instances"
            )
        else:
            self._train_instances = test_dataset

        # create dataloader
        test_dataloader = create_dataloader(
            self._train_instances,
            batch_size=batch_size or self._trainer_config.batch_size,
            num_workers=self._trainer_config.num_workers_for_dataloader,
        )

        metrics = self.trainer.test(
            self._pipeline.model, test_dataloaders=test_dataloader, verbose=verbose
        )[0]

        if output_dir:
            with (output_dir / "metrics.json").open("w") as file:
                json.dump(sanitize(metrics), file)

        return metrics


class ModelCheckpointWithVocab(ModelCheckpoint):
    def on_pretrain_routine_start(self, trainer, pl_module: "PipelineModel"):
        super().on_pretrain_routine_start(trainer, pl_module)
        if os.path.isdir(self.dirpath):
            pl_module.vocab.save_to_files(os.path.join(self.dirpath, "vocabulary"))


def create_dataloader(
    instance_dataset: InstanceDataset,
    batch_size: int = 16,
    data_bucketing: bool = False,
    num_workers: int = 0,
) -> PyTorchDataLoader:
    """Returns a pytorch DataLoader for AllenNLP instances

    Parameters
    ----------
    instance_dataset
        The dataset of instances for the DataLoader
    batch_size
        Batch size
    data_bucketing
        If True, tries to sort batches with respect to the maximum input lengths per batch.
        Not supported for lazily loaded data!
    num_workers
        How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        Default: 0

    Returns
    -------
    data_loader
    """
    if data_bucketing and isinstance(instance_dataset, IterableDataset):
        _LOGGER.warning(
            "'data_bucketing' is not supported for lazily loaded data. We will deactivate it."
        )
        data_bucketing = False

    return PyTorchDataLoader(
        instance_dataset,
        batch_size=1 if data_bucketing else batch_size,
        batch_sampler=BucketBatchSampler(
            data_source=instance_dataset,
            batch_size=batch_size,
        )
        if data_bucketing
        else None,
        num_workers=num_workers,
    )
