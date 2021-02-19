import inspect
import json
import logging
import os
import pickle
import re
import warnings
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy
import pytorch_lightning as pl
import torch
from allennlp.common import Params
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import remove_keys_from_params
from allennlp.common.util import sanitize
from allennlp.data import Batch
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.models.model import _DEFAULT_WEIGHTS
from allennlp.nn import util

from . import vocabulary
from .backbone import ModelBackbone
from .configuration import PipelineConfiguration
from .configuration import PredictionConfiguration
from .featurizer import FeaturizeError
from .helpers import split_signature_params_by_predicate
from .modules.heads import TaskHead
from .modules.heads import TaskPrediction


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


class PipelineModel(pl.LightningModule, Registrable):
    """
    This class represents pipeline model implementation for connect biome.text concepts with
    allennlp implementation details

    This class manages the head + backbone encoder, keeping the allennlnlp Model lifecycle. This class
    should be hidden to api users.

    Parameters
    ----------
    name
        Name of the pipeline model
    head
        TaskHead of the pipeline model

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
    _LOGGER = logging.getLogger(__name__)

    def __init__(self, name: str, head: TaskHead):
        super().__init__()

        self.name = name
        self.vocab = head.backbone.vocab
        self._head = None
        self.set_head(head)

        self.file_path: Optional[str] = None

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
        cls: Type["PipelineModel"],
        params: Params,
        vocab: Optional[Vocabulary] = None,
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

        Returns
        -------
        pipeline_model
        """

        config = params.pop("config")
        if not isinstance(config, PipelineConfiguration):
            config = PipelineConfiguration.from_params(config)

        vocab = vocab or vocabulary.create_empty_vocabulary()
        tokenizer = config.build_tokenizer()
        featurizer = config.features.compile_featurizer(tokenizer)
        embedder = config.build_embedder(vocab)

        return cls(
            name=config.name,
            head=config.head.compile(
                backbone=ModelBackbone(
                    vocab,
                    featurizer=featurizer,
                    embedder=embedder,
                    encoder=config.encoder,
                )
            ),
        )

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
        return self._head.forward(*args, **kwargs)

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

        Parameters
        ----------
        batch
            A list of dictionaries that represents a batch of inputs.
            The dictionary keys must comply with the `self.inputs` attribute.
        prediction_config
            Contains configurations for the prediction
        """
        if self.training:
            warnings.warn(
                "Training mode enabled."
                "Disabling training mode automatically. You can manually disable it: "
                "self.eval() or self.train(False)"
            )
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

    def extend_embedder_vocab(
        self, embedding_sources_mapping: Dict[str, str] = None
    ) -> None:
        """Iterates through all embedding modules and assures it can embed the extended vocab.

        This is required in fine-tuning or transfer learning scenarios where the model was trained with
        the original vocabulary, but during fine-tuning/transfer-learning, the vocabulary was extended
        (original + new-data vocabulary).

        Parameters
        ----------
        embedding_sources_mapping
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
                module.extend_vocab(
                    self.vocab,
                    extension_pretrained_file=pretrained_file,
                    model_path=model_path,
                )

    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        """Needed by the AllenNLP trainer.

        We have not implemented gegularization in our PipelineModel at the moment, so this method does nothing.
        """
        pass

    @classmethod
    def load(
        cls,
        config: Params,
        serialization_dir: Union[str, PathLike],
        weights_file: Optional[Union[str, PathLike]] = None,
        cuda_device: int = -1,
    ) -> "PipelineModel":
        """Instantiates an already-trained model, based on the experiment configuration and some optional overrides.

        Parameters
        ----------
        config
            The configuration of the model
        serialization_dir
            Path to the directory where the model is serialized
        weights_file
            File with the weights of the model.
            By default we load the _DEFAULT_WEIGHTS file inside the serialization dir
        cuda_device
            Device on which to put the model

        Returns
        -------
        pipeline_model
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice(
            "type", Vocabulary.list_available(), True
        )
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        model_params = config.get("model")

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings/weights from. We're now _loading_ the model, so those weights will already be
        # stored in our model. We don't need any pretrained weight file or initializers anymore,
        # and we don't want the code to look for it, so we remove it from the parameters here.
        remove_keys_from_params(model_params)
        model = PipelineModel.from_params(vocab=vocab, params=model_params)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        # If vocab+embedding extension was done, the model initialized from from_params
        # and one defined by state dict in weights_file might not have same embedding shapes.
        # Eg. when model embedder module was transferred along with vocab extension, the
        # initialized embedding weight shape would be smaller than one in the state_dict.
        # So calling model embedding extension is required before load_state_dict.
        # If vocab and model embeddings are in sync, following would be just a no-op.
        model.extend_embedder_vocab()

        # Load state dict. We pass `strict=False` so PyTorch doesn't raise a RuntimeError
        # if the state dict is missing keys because we handle this case below.
        model_state = torch.load(
            weights_file, map_location=util.device_mapping(cuda_device)
        )
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        # Modules might define a class variable called `authorized_missing_keys`,
        # a list of regex patterns, that tells us to ignore missing keys that match
        # any of the patterns.
        # We sometimes need this in order to load older models with newer versions of AllenNLP.

        def filter_out_authorized_missing_keys(module, prefix=""):
            nonlocal missing_keys
            for pat in getattr(module.__class__, "authorized_missing_keys", None) or []:
                missing_keys = [
                    k
                    for k in missing_keys
                    if k.startswith(prefix) and re.search(pat[len(prefix) :], k) is None
                ]
            for name, child in module._modules.items():
                if child is not None:
                    filter_out_authorized_missing_keys(child, prefix + name + ".")

        filter_out_authorized_missing_keys(model)

        if unexpected_keys or missing_keys:
            raise RuntimeError(
                f"Error loading state dict for {model.__class__.__name__}\n\t"
                f"Missing keys: {missing_keys}\n\t"
                f"Unexpected keys: {unexpected_keys}"
            )

        return model

    def forward_on_instances(
        self, instances: List[Instance]
    ) -> List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of `Instances`, converts that text into arrays using this model's `Vocabulary`,
        passes those arrays through `self.forward()` and returns the result.

        Before returning the result, we convert any `torch.Tensors` into numpy arrays and
        separate the batched output into a list of individual dicts per instance.

        Parameters
        ----------
        instances
            The instances to run the model on.

        Returns
        -------
        A list of the models output for each instance.
        """
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self(**model_input)

            instance_separated_output: List[Dict[str, numpy.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(
                    instance_separated_output, output
                ):
                    instance_output[name] = batch_element
            return instance_separated_output

    def _get_prediction_device(self) -> int:
        """
        This method checks the device of the model parameters to determine the cuda_device
        this model should be run on for predictions.  If there are no parameters, it returns -1.

        Returns
        -------
        cuda_device_or_cpu
        """
        devices = {util.get_device_of(param) for param in self.parameters()}

        if len(devices) > 1:
            devices_string = ", ".join(str(x) for x in devices)
            raise ConfigurationError(
                f"Parameters have mismatching cuda_devices: {devices_string}"
            )
        elif len(devices) == 1:
            return devices.pop()
        else:
            return -1

    def _maybe_warn_for_unseparable_batches(self, output_key: str):
        """
        This method warns once if a user implements a model which returns a dictionary with
        values which we are unable to split back up into elements of the batch. This is controlled
        by a class attribute `_warn_for_unseperable_batches` because it would be extremely verbose
        otherwise.
        """
        if output_key not in self._warn_for_unseparable_batches:
            self._LOGGER.warning(
                f"Encountered the {output_key} key in the model's return dictionary which "
                "couldn't be split by the batch size. Key will be ignored."
            )
            # We only want to warn once for this key,
            # so we set this to false so we don't warn again.
            self._warn_for_unseparable_batches.add(output_key)


PipelineModel.register(PipelineModel.__name__, exist_ok=True)(PipelineModel)
