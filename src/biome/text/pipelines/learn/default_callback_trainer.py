"""This module includes the default biome callback trainer and some extra functions/classes for this purpose"""
import logging

import torch
from allennlp.common import Params
from allennlp.common.checks import parse_cuda_device
from allennlp.data import Instance, DataIterator
from allennlp.models import Model
from allennlp.training import CallbackTrainer, TrainerBase
from allennlp.training.callbacks import (
    Callback,
    GradientNormAndClip,
    Checkpoint,
    TrackMetrics,
    Validate,
    LogToTensorboard,
)
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_pieces import TrainerPieces
from typing import Iterable, Optional, Union, List

from .callbacks import EvaluateCallback, LoggingCallback


class DefaultCallbackTrainer(CallbackTrainer):
    """
    An callback trainer with some extra callbacks already configured
    """

    _LOGGER = logging.getLogger(__name__)

    @staticmethod
    def _callbacks_configuration(
        patience: Optional[int],
        validation_metric: str,
        num_serialized_models_to_keep: int,
        should_log_learning_rate: bool,
    ) -> List[dict]:

        return [
            dict(type="gradient_norm_and_clip", grad_norm=True),
            dict(
                type="checkpoint",
                checkpointer={
                    "num_serialized_models_to_keep": num_serialized_models_to_keep
                },
            ),
            dict(
                type="track_metrics",
                patience=patience,
                validation_metric=validation_metric,
            ),
            "validate",
            dict(
                type="log_to_tensorboard",
                log_batch_size_period=10,
                should_log_learning_rate=should_log_learning_rate,
            ),  # TODO parameterize
            "evaluate",
            "logging",
        ]

    @classmethod
    def from_params(
        cls,  # type: ignore
        params: Params,
        serialization_dir: str,
        recover: bool = False,
        cache_directory: str = None,
        cache_prefix: str = None,
    ) -> CallbackTrainer:
        trainer_params = params["trainer"]

        patience = trainer_params.pop("patience", None)
        validation_metric = trainer_params.pop("validation_metric", "-loss")
        num_serialized_models_to_keep = trainer_params.pop(
            "num_serialized_models_to_keep", 1
        )
        should_log_learning_rate = trainer_params.pop("should_log_learning_rate", False)

        trainer_params["callbacks"] = cls._callbacks_configuration(
            patience,
            validation_metric,
            num_serialized_models_to_keep,
            should_log_learning_rate,
        )
        return CallbackTrainer.from_params(
            params, serialization_dir, recover, cache_directory, cache_prefix
        )


TrainerBase.register(DefaultCallbackTrainer.__name__, exist_ok=True)(
    DefaultCallbackTrainer
)
