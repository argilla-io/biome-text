import inspect
import json
import logging
import os
import pickle
import warnings
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import allennlp
import pytorch_lightning as pl
import torch
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.models.archival import CONFIG_NAME

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.configuration import PipelineConfiguration
from biome.text.configuration import PredictionConfiguration
from biome.text.featurizer import FeaturizeError
from biome.text.helpers import split_signature_params_by_predicate
from biome.text.modules.heads import TaskHead
from biome.text.modules.heads import TaskPrediction


class _HashDict(dict):
    """
    Hashable dict implementation.
    BE AWARE! Since dicts are mutable, the hash can change!
    """

    def __hash__(self):
        return pickle.dumps(self).__hash__()


class _HashList(list):
    """
    Hashable list implementation.
    BE AWARE! Since lists are mutable, the hash can change!
    """

    def __hash__(self):
        return pickle.dumps(self).__hash__()


class PipelineModel(allennlp.models.Model, pl.LightningModule):
    """
    This class represents pipeline model implementation for connect biome.text concepts with
    allennlp implementation details

    This class manages the head + backbone encoder, keeping the allennlnlp Model lifecycle. This class
    should be hidden to api users.

    Parameters
    ----------
    config
        Configuration of the pipeline
    vocab
        The vocabulary of the pipeline. If None, an empty vocabulary will be created (default).

    Attributes
    ----------
    name: str
        Name of the pipeline model
    head: TaskHead
        TaskHead of the pipeline model
    vocab: Vocabulary
        The vocabulary of the model, comes from allennlp.models.Model
    file_path: Optional[str]
        File path to a serialized version of this pipeline model
    inputs: List[str]
        The model inputs
    output: List[str]
        The model outputs (not prediction): Corresponding to the `TaskHead.featurize` optional arguments.
    """

    PREDICTION_FILE_NAME = "predictions.json"
    TRAINING_METRICS_PREFIX = "training"
    VALIDATION_METRICS_PREFIX = "validation"
    TEST_METRICS_PREFIX = "test"
    _LOGGER = logging.getLogger(__name__)

    def __init__(self, config: Dict, vocab: Optional[Vocabulary] = None):
        super().__init__(vocab=vocab or vocabulary.create_empty_vocabulary())

        # saves the config in the pl checkpoints
        self.save_hyperparameters("config")

        config = PipelineConfiguration.from_dict(config)
        tokenizer = config.build_tokenizer()
        featurizer = config.features.compile_featurizer(tokenizer)
        embedder = config.build_embedder(self.vocab)
        head = config.head.compile(
            backbone=ModelBackbone(
                self.vocab,
                featurizer=featurizer,
                embedder=embedder,
                encoder=config.encoder,
            )
        )

        self.name = config.name
        self._head = None
        self.set_head(head)

        self.file_path: Optional[str] = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        # The lr_scheduler dict follows the Lightning format:
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#learning-rate-scheduling
        self.lr_scheduler: Optional[Dict] = None

        self.best_metrics: Optional[Dict[str, torch.Tensor]] = None
        # This is set by our trainer to figure out the best_metrics
        # what metric to monitor?
        self.monitor: Optional[str] = None
        # shall the metric increase ("max") or decrease ("min")?
        self.monitor_mode: Optional[str] = None

    def _update_head_related_attributes(self):
        """Updates the inputs/outputs and default mapping attributes, calculated from model head"""
        required, optional = split_signature_params_by_predicate(
            self._head.featurize, lambda p: p.default == inspect.Parameter.empty
        )
        self._inputs = self._head.inputs() or [p.name for p in required]
        self._output = [p.name for p in optional if p.name not in self._inputs] or [
            None
        ]

    @classmethod
    def from_params(
        cls: "PipelineModel",
        params: Params,
        vocab: Optional[Vocabulary] = None,
        **extras,
    ) -> "PipelineModel":
        """
        Load the model implementation from params. We build manually each component from config sections.

        The param keys matches exactly with keys in yaml configuration files

        Parameters
        ----------
        params
            The config key in these params is used to build the model components
        vocab
            The vocabulary for the model
        **extras
            Necessary for AllenNLP from_params machinery

        Returns
        -------
        pipeline_model
        """

        config = params.pop("config", keep_as_dict=True)
        return cls(config=config, vocab=vocab)

    @property
    def head(self) -> TaskHead:
        """Get the model head"""
        return self._head

    def set_head(self, head: TaskHead) -> None:
        """Set a head and update related model attributes"""
        self._head = head
        self._update_head_related_attributes()

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """The main forward method just wraps the head forward method"""
        return self._head(*args, **kwargs)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Fetch metrics defined in head layer"""
        return self._head.get_metrics(reset)

    def text_to_instance(self, **inputs) -> Optional[Instance]:
        """Applies the head featurize method"""
        try:
            return self._head.featurize(**inputs)
        except FeaturizeError as error:
            # we cannot featurize the input (empty strings, etc.)
            self._LOGGER.warning(error)
        except TypeError as error:
            # probably wrong input arguments for the head
            raise TypeError(
                f"Please check your input arguments, expected: {self.inputs}, actual: {inputs.keys()}"
            ) from error

    def extend_vocabulary(self, vocab: Vocabulary):
        """Extend the model's vocabulary with `vocab`

        Parameters
        ----------
        vocab
            The model's vocabulary will be extended with this one.
        """
        # self.vocab and self._head.backbone.vocab point to the same vocab!
        self.vocab.extend_from_vocab(vocab)

        # updates the embedding matrices
        self.extend_embedder_vocab()
        # updates head specific things
        self._head.on_vocab_update()

    def extend_embedder_vocab(
        self, embedding_sources_mapping: Dict[str, str] = None
    ) -> None:
        """
        Iterates through all embedding modules in the model and assures it can embed
        with the extended vocab. This is required in fine-tuning or transfer learning
        scenarios where model was trained with original vocabulary but during
        fine-tuning/transfer-learning, it will have it work with extended vocabulary
        (original + new-data vocabulary).

        # Parameters

        embedding_sources_mapping : `Dict[str, str]`, optional (default = `None`)
            Mapping from model_path to pretrained-file path of the embedding
            modules. If pretrained-file used at time of embedding initialization
            isn't available now, user should pass this mapping. Model path is
            path traversing the model attributes upto this embedding module.
            Eg. "_text_field_embedder.token_embedder_tokens".
        """
        # self.named_modules() gives all sub-modules (including nested children)
        # The path nesting is already separated by ".": eg. parent_module_name.child_module_name
        embedding_sources_mapping = embedding_sources_mapping or {}
        for model_path, module in self.named_modules():
            if hasattr(module, "extend_vocab"):
                pretrained_file = embedding_sources_mapping.get(model_path)
                # Show useful information when reading from a pretrained file, kind of an ugly hack
                if module._pretrained_file is not None:
                    original_logging_level = logging.getLogger("allennlp").level
                    logging.getLogger("allennlp").setLevel("INFO")

                module.extend_vocab(
                    self.vocab,
                    extension_pretrained_file=pretrained_file,
                    model_path=model_path,
                )

                if module._pretrained_file is not None:
                    logging.getLogger("allennlp").setLevel(original_logging_level)

    @property
    def inputs(self) -> List[str]:
        """The model inputs. Corresponding to head.featurize required argument names"""
        return self._inputs

    @property
    def output(self) -> List[str]:
        """The model outputs (not prediction): Corresponding to head.featurize optional argument names."""
        return self._output

    def init_prediction_logger(
        self, output_dir: str, max_bytes: int = 20000000, backup_count: int = 20
    ):
        """Initialize the prediction logger.

        If initialized we will log all predictions to a file called *predictions.json* in the `output_folder`.

        Parameters
        ----------
        output_dir
            Path to the folder in which we create the *predictions.json* file.
        max_bytes
            Passed on to logging.handlers.RotatingFileHandler
        backup_count
            Passed on to logging.handlers.RotatingFileHandler

        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        predictions_logger = logging.getLogger(output_dir)
        predictions_logger.setLevel(logging.DEBUG)
        # This flag avoids logging messages to be propagated to the parent loggers
        predictions_logger.propagate = False
        file_handler = RotatingFileHandler(
            os.path.join(output_dir, self.PREDICTION_FILE_NAME),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(logging.INFO)
        predictions_logger.addHandler(file_handler)

        setattr(self, "_prediction_logger", predictions_logger)

    def init_prediction_cache(self, max_size: int) -> None:
        """Initialize a prediction cache using the functools.lru_cache decorator

        Parameters
        ----------
        max_size
            Save up to max_size most recent items.
        """

        if hasattr(self, "_predict_with_cache"):
            warnings.warn(
                "Prediction cache already initiated!", category=RuntimeWarning
            )
            return

        predict_with_cache = lru_cache(maxsize=max_size)(self.predict)

        def predict_wrapper(*args, **kwargs):
            def hashable_value(value) -> Any:
                if isinstance(value, dict):
                    return _HashDict(value)
                if isinstance(value, (list, tuple)):
                    return _HashList(value)
                return value

            return predict_with_cache(
                *[hashable_value(arg_value) for arg_value in args],
                **{
                    key: hashable_value(input_value)
                    for key, input_value in kwargs.items()
                },
            )

        self.__setattr__("predict", predict_wrapper)
        self.__setattr__("_predict_with_cache", predict_with_cache)

    def _log_predictions(
        self,
        batch: Iterable[Dict[str, Any]],
        predictions: Iterable[TaskPrediction],
    ) -> None:
        """Log predictions to a file for a model analysis and feedback sessions.

        Parameters
        ----------
        batch
            Input data to the model
        predictions
            Returned predictions from the model
        """
        for input_dict, prediction in zip(batch, predictions):
            self._prediction_logger.info(
                json.dumps(
                    dict(inputs=input_dict, prediction=sanitize(prediction.as_dict()))
                )
            )

    def predict(
        self,
        batch: List[Dict[str, Union[str, List[str], Dict[str, str]]]],
        prediction_config: PredictionConfiguration,
    ) -> List[Optional[TaskPrediction]]:
        """Returns predictions given some input data based on the current state of the model

        The keys of the input dicts in the batch must coincide with the `self.inputs` attribute.
        TODO: Comply with LightningModule API + Trainer API (means move instance creation logic to Pipeline)

        Parameters
        ----------
        batch
            A list of dictionaries that represents a batch of inputs.
            The dictionary keys must comply with the `self.inputs` attribute.
        prediction_config
            Contains configurations for the prediction
        """
        if self.training:
            self.eval()

        instances = [self.text_to_instance(**input_dict) for input_dict in batch]
        # Filter out None instances, that is when the head could not create an instance out of the input
        none_indices, not_none_instances = [], []
        for i, instance in enumerate(instances):
            if instance is None:
                none_indices.append(i)
            else:
                not_none_instances.append(instance)

        try:
            forward_outputs = self.forward_on_instances(not_none_instances)
        except Exception as error:
            input_examples = [
                example for i, example in enumerate(batch) if i not in none_indices
            ]
            self._LOGGER.exception(error)
            self._LOGGER.warning(
                f"Failed to make a forward pass for '{input_examples}'"
            )
            return [None] * len(batch)

        predictions = []
        for forward_output, instance in zip(forward_outputs, not_none_instances):
            try:
                predictions.append(
                    self.head.make_task_prediction(
                        forward_output, instance, prediction_config
                    )
                )
            except Exception as error:
                self._LOGGER.exception(error)
                self._LOGGER.warning(
                    f"Failed to make a task prediction for '{forward_output, instance}'"
                )
                predictions.append(None)

        # Add None for the none instances
        for index in none_indices:
            predictions.insert(index, None)

        # Log predictions if the prediction logger was initialized
        if hasattr(self, "_prediction_logger"):
            self._log_predictions(batch, predictions)

        return predictions

    def on_fit_start(self) -> None:
        # Reset metrics
        self.best_metrics = None

    def training_step(self, batch, batch_idx) -> Dict:
        output = self(**batch)
        self.log(
            "training_loss",
            output["loss"],
            on_step=True,
            prog_bar=False,
            on_epoch=False,
        )

        metrics = self.get_metrics()
        for key, val in metrics.items():
            self.log(
                ("training" if key.startswith("_") else "training_") + key,
                val,
                on_step=True,
                prog_bar=not key.startswith("_"),
                on_epoch=False,
            )

        return output

    def training_epoch_end(self, outputs: List[Any]) -> None:
        metrics = self.get_metrics(reset=True)
        for key, val in metrics.items():
            if key.startswith("_"):
                metric_name = self.TRAINING_METRICS_PREFIX + key
            else:
                metric_name = self.TRAINING_METRICS_PREFIX + "_" + key
            self.log(
                metric_name,
                val,
                on_step=False,
                prog_bar=False,
                on_epoch=True,
            )

    def validation_step(self, batch, batch_idx) -> Dict:
        output = self(**batch)
        self.get_metrics()

        return output

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # we do not want to log any metrics for the sanity check
        if self.trainer.sanity_checking:
            return

        # we keep track of the logged metrics to figure out the best metrics
        logged_metrics = {}

        averaged_epoch_loss = sum([output["loss"] for output in outputs]) / len(outputs)
        self.log(
            f"{self.VALIDATION_METRICS_PREFIX}_loss",
            averaged_epoch_loss,
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        logged_metrics[f"{self.VALIDATION_METRICS_PREFIX}_loss"] = averaged_epoch_loss

        metrics = self.get_metrics(reset=True)
        for key, val in metrics.items():
            if key.startswith("_"):
                metric_name = self.VALIDATION_METRICS_PREFIX + key
            else:
                metric_name = self.VALIDATION_METRICS_PREFIX + "_" + key
            self.log(
                metric_name,
                val,
                on_step=False,
                prog_bar=not key.startswith("_"),
                on_epoch=True,
            )
            logged_metrics[metric_name] = val

        # log best metrics
        logged_metrics["epoch"] = self.current_epoch
        if self.best_metrics is None:
            self.best_metrics = logged_metrics
        elif (
            self.monitor_mode == "max"
            and self.best_metrics[self.monitor] < logged_metrics[self.monitor]
        ):
            self.best_metrics = logged_metrics
        elif (
            self.monitor_mode == "min"
            and self.best_metrics[self.monitor] > logged_metrics[self.monitor]
        ):
            self.best_metrics = logged_metrics

        self.log_dict(
            {f"best_{key}": value for key, value in self.best_metrics.items()},
            on_step=False,
            prog_bar=False,
            on_epoch=True,
        )

    def test_step(self, *args, **kwargs) -> Dict:
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        averaged_epoch_loss = sum([output["loss"] for output in outputs]) / len(outputs)
        self.log(
            f"{self.TEST_METRICS_PREFIX}_loss",
            averaged_epoch_loss,
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )

        metrics = self.get_metrics(reset=True)
        for key, val in metrics.items():
            if key.startswith("_"):
                metric_name = self.TEST_METRICS_PREFIX + key
            else:
                metric_name = self.TEST_METRICS_PREFIX + "_" + key
            self.log(
                metric_name,
                val,
                on_step=False,
                prog_bar=not key.startswith("_"),
                on_epoch=True,
            )

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        return [self.optimizer], [self.lr_scheduler]


allennlp.models.Model.register(PipelineModel.__name__, exist_ok=True)(PipelineModel)
