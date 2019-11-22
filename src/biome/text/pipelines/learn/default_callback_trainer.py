"""This module includes the default biome callback trainer and some extra functions/classes for this purpose"""
import logging

from allennlp.common import Params
from allennlp.training import CallbackTrainer, TrainerBase
from typing import Optional, List

from .callbacks import LoggingCallback, EvaluateCallback

__alias__ = [LoggingCallback, EvaluateCallback]


class DefaultCallbackTrainer(CallbackTrainer):
    """
    An callback trainer with some extra callbacks already configured
    """

    _LOGGER = logging.getLogger(__name__)

    @staticmethod
    def _callbacks_configuration(trainer_params: Params) -> List[dict]:

        patience = trainer_params.pop("patience", None)
        validation_metric = trainer_params.pop("validation_metric", "-loss")
        num_serialized_models_to_keep = trainer_params.pop(
            "num_serialized_models_to_keep", 1
        )
        should_log_learning_rate = trainer_params.pop("should_log_learning_rate", False)
        summary_interval = trainer_params.pop("summary_interval", 100)
        learning_rate_scheduler_params = trainer_params.pop(
            "learning_rate_scheduler", None
        )
        if learning_rate_scheduler_params:
            learning_rate_scheduler_params = learning_rate_scheduler_params.as_dict()

        callbacks = [
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
                summary_interval=summary_interval,
            ),  # TODO parameterize
            "evaluate",
            "logging",
        ]

        if learning_rate_scheduler_params:
            callbacks.append(
                dict(
                    type="update_learning_rate",
                    learning_rate_scheduler=learning_rate_scheduler_params,
                )
            )

        return callbacks

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

        trainer_params["callbacks"] = cls._callbacks_configuration(trainer_params)

        return CallbackTrainer.from_params(
            params, serialization_dir, recover, cache_directory, cache_prefix
        )


TrainerBase.register(DefaultCallbackTrainer.__name__, exist_ok=True)(
    DefaultCallbackTrainer
)
