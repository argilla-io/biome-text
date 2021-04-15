import logging
import math
import multiprocessing
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from allennlp.common import Params
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

from biome.text.configuration import LightningTrainerConfiguration
from biome.text.configuration import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.dataset import InstanceDataset
from biome.text.pipeline import Pipeline

if TYPE_CHECKING:
    from biome.text.model import PipelineModel

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
    """A class for training a `biome.text.Pipeline`.

    It is basically a light wrapper around the awesome Pytorch Lightning Trainer to define custom defaults and
    facilitate the interaction with our pipelines.

    Parameters
    ----------
    pipeline
        Pipeline to train
    train_dataset
        The training dataset
    valid_dataset
        The validation dataset. Default: `None`.
    trainer_config
        The configuration of the trainer. Default: `LightningTrainerConfiguration()`.
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
        pipeline: Pipeline,
        train_dataset: Union[Dataset, InstanceDataset],
        valid_dataset: Optional[Union[Dataset, InstanceDataset]] = None,
        trainer_config: Optional[LightningTrainerConfiguration] = None,
        vocab_config: Optional[Union[str, VocabularyConfiguration]] = "default",
        lazy: bool = False,
    ):
        self._pipeline = pipeline
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        # since we will make changes to the config, better to make a copy -> asdict returns a deep copy
        self._trainer_config = (
            LightningTrainerConfiguration(**asdict(trainer_config))
            if trainer_config is not None
            else LightningTrainerConfiguration()
        )
        self._vocab_config: Optional[VocabularyConfiguration] = (
            VocabularyConfiguration() if vocab_config == "default" else vocab_config
        )
        self._lazy = lazy

        # we give some special attention to these loggers/callbacks
        self._wandb_logger: Optional[WandbLogger] = None
        self._model_checkpoint: Optional[ModelCheckpoint] = None

        # add default callbacks/loggers/gpu
        self._trainer_config.callbacks = self._add_default_callbacks()
        if self._trainer_config.logger is not False:
            self._trainer_config.logger = self._add_default_loggers()
        if self._trainer_config.gpus is None and torch.cuda.is_available():
            self._trainer_config.gpus = 1

        # create optimizer
        self._pipeline.model.optimizer = Optimizer.from_params(
            Params(
                {
                    "model_parameters": self._pipeline.model.named_parameters(),
                    **self._trainer_config.optimizer,
                }
            )
        )

        # create lr scheduler
        if not (
            self._trainer_config.warmup_steps == 0
            and self._trainer_config.lr_decay is None
        ):
            self._pipeline.model.lr_scheduler = self._create_lr_scheduler()

        self.trainer = pl.Trainer(**self._trainer_config.lightning_params)

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
            monitor = self._trainer_config.monitor if self._valid_dataset else None
            mode = self._trainer_config.monitor_mode
            save_top_k = (
                self._trainer_config.save_top_k_checkpoints
                if self._valid_dataset
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
            and self._valid_dataset is not None
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
            len(self._train_dataset) / self._trainer_config.batch_size
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

        Parameters
        ----------
        output_dir
            If specified, save the trained pipeline to this directory. Default: 'output'.
        exist_ok
            If True, overwrite the content of `output_dir`. Default: False.
        """
        try:
            output_dir = Path(output_dir)
        except TypeError:
            pass
        else:
            output_dir.mkdir(exist_ok=exist_ok)

        # create instances
        if isinstance(self._train_dataset, Dataset):
            train_instances = self._train_dataset.to_instances(
                self._pipeline, lazy=self._lazy
            )
        else:
            train_instances = self._train_dataset

        if isinstance(self._valid_dataset, Dataset):
            valid_instances = self._valid_dataset.to_instances(
                self._pipeline, lazy=self._lazy
            )
        else:
            valid_instances = self._valid_dataset

        # create vocab
        vocab_config = (
            VocabularyConfiguration()
            if self._vocab_config == "default"
            else self._vocab_config
        )
        if vocab_config is not None:
            vocab_datasets = [train_instances]
            if valid_instances is not None and self._vocab_config.include_valid_data:
                vocab_datasets += [valid_instances]
            self._pipeline.create_vocab(vocab_datasets, config=vocab_config)

        # create dataloaders
        train_dataloader = create_dataloader(
            train_instances,
            batch_size=self._trainer_config.batch_size,
            data_bucketing=self._trainer_config.data_bucketing,
        )
        valid_dataloader = (
            create_dataloader(
                self._valid_dataset.to_instances(self._pipeline, lazy=self._lazy),
                batch_size=self._trainer_config.batch_size,
                data_bucketing=self._trainer_config.data_bucketing,
            )
            if self._valid_dataset is not None
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

    def _load_best_weights(self):
        """Load weights from the best model checkpoint"""
        checkpoint_path = self._model_checkpoint.best_model_path
        if checkpoint_path:
            _LOGGER.info("Loading best weights ...")
            checkpoint = pl_load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
            self._pipeline.model.load_state_dict(checkpoint["state_dict"])


class ModelCheckpointWithVocab(ModelCheckpoint):
    def on_pretrain_routine_start(self, trainer, pl_module: "PipelineModel"):
        super().on_pretrain_routine_start(trainer, pl_module)
        if os.path.isdir(self.dirpath):
            pl_module.vocab.save_to_files(os.path.join(self.dirpath, "vocabulary"))


def create_dataloader(
    instance_dataset: InstanceDataset,
    batch_size: int = 16,
    data_bucketing: bool = False,
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
        num_workers=multiprocessing.cpu_count(),
    )
