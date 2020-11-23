import copy
import inspect
import logging
import os
import shutil
import tempfile
from biome.text import vocabulary
from biome.text.configuration import FindLRConfiguration
from biome.text.configuration import PipelineConfiguration
from biome.text.configuration import TrainerConfiguration
from biome.text.configuration import VocabularyConfiguration
from biome.text.dataset import InstancesDataset
from biome.text.dataset import Dataset
from biome.text.errors import ActionNotSupportedError
from biome.text.errors import EmptyVocabError
from biome.text.features import TransformersFeatures
from biome.text.helpers import update_method_signature
from inspect import Parameter
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy
import torch
from allennlp.commands.find_learning_rate import search_learning_rate
from allennlp.common import Params
from allennlp.data import AllennlpLazyDataset
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.models import load_archive
from allennlp.models.archival import Archive
from dask.dataframe import DataFrame

from ._model import PipelineModel
from .backbone import ModelBackbone
from .loggers import add_default_wandb_logger_if_needed
from .loggers import BaseTrainLogger
from .modules.heads import TaskHead
from .modules.heads import TaskHeadConfiguration
from .training_results import TrainingResults

try:
    import ujson as json
except ModuleNotFoundError:
    import json


logging.getLogger("allennlp").setLevel(logging.ERROR)
logging.getLogger("elasticsearch").setLevel(logging.ERROR)


class Pipeline:
    """Manages NLP models configuration and actions.

    Use `Pipeline` for creating new models from a configuration or loading a pre-trained model.

    Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.
    """

    __LOGGER = logging.getLogger(__name__)
    _model: PipelineModel = None
    _config: PipelineConfiguration = None

    @classmethod
    def from_yaml(cls, path: str, vocab_path: Optional[str] = None) -> "Pipeline":
        """Creates a pipeline from a config yaml file

        Parameters
        ----------
        path : `str`
            The path to a YAML configuration file
        vocab_path : `Optional[str]`
            If provided, the pipeline vocab will be loaded from this path

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """
        pipeline_configuration = PipelineConfiguration.from_yaml(path)

        return cls.from_config(pipeline_configuration, vocab_path=vocab_path)

    @classmethod
    def from_config(
        cls,
        config: Union[PipelineConfiguration, dict],
        vocab_path: Optional[str] = None,
    ) -> "_BlankPipeline":
        """Creates a pipeline from a `PipelineConfiguration` object or a configuration dictionary

        Parameters
        ----------
        config: `Union[PipelineConfiguration, dict]`
            A `PipelineConfiguration` object or a configuration dict
        vocab_path: `Optional[str]`
            If provided, the pipeline vocabulary will be loaded from this path

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """
        if isinstance(config, dict):
            config = PipelineConfiguration.from_dict(config)
        return _BlankPipeline(
            config=config, vocab=vocabulary.load_vocabulary(vocab_path)
        )

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "_PreTrainedPipeline":
        """Loads a pipeline from a pre-trained pipeline providing a *model.tar.gz* file path

        Parameters
        ----------
        path: `str`
            The path to the *model.tar.gz* file of a pre-trained `Pipeline`

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """
        return _PreTrainedPipeline(pretrained_path=path, **kwargs)

    def init_prediction_logger(self, output_dir: str, max_logging_size: int = 100):
        """Initializes the prediction logging.

        If initialized, all predictions will be logged to a file called *predictions.json* in the `output_dir`.

        Parameters
        ----------
        output_dir: str
            Path to the folder in which we create the *predictions.json* file.
        max_logging_size: int
            Max disk size to use for prediction logs
        """
        max_bytes = max_logging_size * 1000000
        max_bytes_per_file = 2000000
        n_backups = int(max_bytes / max_bytes_per_file)
        self._model.init_prediction_logger(
            output_dir, max_bytes=max_bytes_per_file, backup_count=n_backups
        )

    def init_prediction_cache(self, max_size: int) -> None:
        """Initializes the cache for input predictions

        Parameters
        ----------
        max_size
            Save up to max_size most recent (inputs).
        """
        self._model.init_prediction_cache(max_size)

    def find_lr(
        self,
        trainer_config: TrainerConfiguration,
        find_lr_config: FindLRConfiguration,
        training_data: Union[Dataset, InstancesDataset],
    ):
        """Returns a learning rate scan on the model.

        It increases the learning rate step by step while recording the losses.
        For a guide on how to select the learning rate please refer to this excellent
        [blog post](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)

        Parameters
        ----------
        trainer_config
            A trainer configuration
        find_lr_config
            A configuration for finding the learning rate
        training_data
            The training data

        Returns
        -------
        (learning_rates, losses)
            Returns a list of learning rates and corresponding losses.
            Note: The losses are recorded before applying the corresponding learning rate
        """
        from biome.text._helpers import create_trainer_for_finding_lr

        if vocabulary.is_empty(self._model.vocab, self.config.features.keys):
            raise EmptyVocabError(
                "Found an empty vocabulary. "
                "You probably forgot to create a vocabulary with '.create_vocabulary()'."
            )

        if isinstance(training_data, Dataset):
            training_data: AllennlpLazyDataset = training_data.to_instances(
                pipeline=self, 
                lazy=True,
            )
        training_data.index_with(self._model.vocab)

        trainer = create_trainer_for_finding_lr(
            model=self._model,
            trainer_config=trainer_config,
            training_data=training_data,
        )

        # restore the state after the lr search is done
        tmp_state_path = (
            Path(tempfile.gettempdir()) / f"model_state_before_findlr_{id(self)}.pth"
        )
        torch.save(self._model.state_dict(), tmp_state_path)
        try:
            learning_rates, losses = search_learning_rate(
                trainer=trainer,
                start_lr=find_lr_config.start_lr,
                end_lr=find_lr_config.end_lr,
                num_batches=find_lr_config.num_batches,
                linear_steps=find_lr_config.linear_steps,
                stopping_factor=find_lr_config.stopping_factor,
            )
        finally:
            self._model.load_state_dict(torch.load(tmp_state_path))

        return learning_rates, losses

    def train(
        self,
        output: str,
        training: Union[Dataset, InstancesDataset],
        trainer: Optional[TrainerConfiguration] = None,
        validation: Optional[Union[Dataset, InstancesDataset]] = None,
        test: Optional[Union[Dataset, InstancesDataset]] = None,
        extend_vocab: Optional[VocabularyConfiguration] = None,
        loggers: List[BaseTrainLogger] = None,
        lazy: bool = False,
        restore: bool = False,
        quiet: bool = False,
    ) -> TrainingResults:
        """Launches a training run with the specified configurations and data sources

        Parameters
        ----------
        output
            The experiment output path
        training
            The training Dataset
        trainer
            The trainer file path
        validation
            The validation Dataset (optional)
        test
            The test Dataset (optional)
        extend_vocab
            Extends the vocabulary tokens with the provided VocabularyConfiguration
        loggers
            A list of loggers that execute a callback before the training, after each epoch,
            and at the end of the training (see `biome.text.logger.MlflowLogger`, for example)
        lazy
            If true, dataset instances are lazily loaded from disk, otherwise they are loaded and kept in memory.
        restore
            If enabled, tries to read previous training status from the `output` folder and
            continues the training process
        quiet
            If enabled, disables most logging messages keeping only warning and error messages.
            In any case, all logging info will be stored into a file at ${output}/train.log

        Returns
        -------
        training_results
            Training results including the generated model path and the related metrics
        """
        if extend_vocab is not None and isinstance(self, _BlankPipeline):
            raise ActionNotSupportedError(
                "If you want to customize pipeline vocab, please use the `create_vocabulary()` method instead"
            )

        trainer = trainer or TrainerConfiguration()
        try:
            if not restore and os.path.isdir(output):
                shutil.rmtree(output)

            self.__configure_training_logging(output, quiet)

            # The original pipeline keeps unchanged
            train_pipeline = self._make_copy()
            vocab = None

            if restore:
                vocab = vocabulary.load_vocabulary(os.path.join(output, "vocabulary"))
            if extend_vocab is not None and not vocab:
                vocab = train_pipeline._extend_vocabulary(
                    train_pipeline.backbone.vocab, vocab_config=extend_vocab
                )
            if vocab:
                train_pipeline._set_vocab(vocab)

            if train_pipeline.has_empty_vocab():
                raise EmptyVocabError(
                    "Found an empty vocabulary. "
                    "You probably forgot to create a vocabulary with '.create_vocabulary()'."
                )

            from ._helpers import PipelineTrainer

            datasets = {"training": training, "validation": validation, "test": test}
            for name, dataset in datasets.items():
                if isinstance(dataset, Dataset):
                    datasets[name] = dataset.to_instances(
                        pipeline=train_pipeline, lazy=lazy
                    )

            loggers = loggers or []
            loggers = add_default_wandb_logger_if_needed(loggers)

            pipeline_trainer = PipelineTrainer(
                train_pipeline,
                trainer_config=trainer,
                output_dir=output,
                epoch_callbacks=loggers,
                **datasets,
            )

            for logger in loggers:
                try:
                    logger.init_train(
                        pipeline=train_pipeline,
                        trainer_configuration=trainer,
                        **datasets,
                    )
                except Exception as e:
                    self.__LOGGER.warning(
                        "Logger %s failed on init_train: %s", logger, e
                    )

            model_path, metrics = pipeline_trainer.train()
            train_results = TrainingResults(model_path, metrics)

            for logger in loggers:
                try:
                    logger.end_train(train_results)
                except Exception as e:
                    self.__LOGGER.warning(
                        "Logger %s failed on end_traing: %s", logger, e
                    )

            return train_results

        finally:
            self.__restore_training_logging()

    def _set_vocab(self, vocab: Vocabulary):
        """
        Updates pipeline vocabulary with passed one. This method will overwrite the current vocab.

        Parameters
        ----------
        vocab:
            The vocabulary to set

        """
        self._model.set_vocab(vocab)

    def _make_copy(self) -> "Pipeline":
        """
        Creates a copy of current pipeline instance
        """
        return _PipelineCopy(self)

    @staticmethod
    def __restore_training_logging():
        """Restore the training logging. This method should be called after a training process"""

        try:
            import tqdm

            tqdm.tqdm.disable = False
        except ModuleNotFoundError:
            pass

        for logger_name, level in [
            ("allennlp", logging.ERROR),
            ("biome", logging.INFO),
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            logger.propagate = True
            logger.handlers = []

    @staticmethod
    def __configure_training_logging(output_dir: str, quiet: bool = False) -> None:
        """Configures training logging"""
        try:
            import tqdm

            tqdm.tqdm.disable = quiet
        except ModuleNotFoundError:
            pass

        os.makedirs(output_dir, exist_ok=True)

        # create file handler which logs even debug messages
        file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
        file_handler.setLevel(logging.INFO)
        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING if quiet else logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # add the handlers to the referred loggers
        for logger_name in ["allennlp", "biome"]:
            logger = logging.getLogger(logger_name)
            logger.propagate = False
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

    def predict(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        """Returns a prediction given some input data based on the current state of the model

        The accepted input is dynamically calculated and can be checked via the `self.inputs` attribute
        (`print(Pipeline.inputs)`)

        Returns
        -------
        predictions: `Dict[str, numpy.ndarray]`
            A dictionary containing the predictions and additional information
        """
        return self._model.predict(*args, **kwargs)

    def predict_batch(
        self, input_dicts: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, numpy.ndarray]]:
        """Returns predictions given some input data based on the current state of the model

        The predictions will be computed batch-wise, which is faster
        than calling `self.predict` for every single input data.

        Parameters
        ----------
        input_dicts
            The input data. The keys of the dicts must comply with the `self.inputs` attribute
        """
        return self._model.predict_batch(input_dicts)

    def explain(self, *args, n_steps: int = 5, **kwargs) -> Dict[str, Any]:
        """Returns a prediction given some input data including the attribution of each token to the prediction.

        The attributions are calculated by means of the [Integrated Gradients](https://arxiv.org/abs/1703.01365) method.

        The accepted input is dynamically calculated and can be checked via the `self.inputs` attribute
        (`print(Pipeline.inputs)`)

        Parameters
        ----------
        n_steps: int
            The number of steps used when calculating the attribution of each token.
            If the number of steps is less than 1, the attributions will not be calculated.

        Returns
        -------
        predictions: `Dict[str, numpy.ndarray]`
            A dictionary containing the predictions and attributions
        """
        return self._model.explain(*args, n_steps=n_steps, **kwargs)

    def explain_batch(
        self, input_dicts: Iterable[Dict[str, Any]], n_steps: int = 5
    ) -> List[Dict[str, numpy.ndarray]]:
        """Returns a prediction given some input data including the attribution of each token to the prediction.

        The predictions will be computed batch-wise, which is faster
        than calling `self.predict` for every single input data.

        The attributions are calculated by means of the [Integrated Gradients](https://arxiv.org/abs/1703.01365) method.

        The accepted input is dynamically calculated and can be checked via the `self.inputs` attribute
        (`print(Pipeline.inputs)`)

        Parameters
        ----------
        input_dicts
            The input data. The keys of the dicts must comply with the `self.inputs` attribute
        n_steps
            The number of steps used when calculating the attribution of each token.
            If the number of steps is less than 1, the attributions will not be calculated.

        Returns
        -------
        predictions
            A list of dictionaries containing the predictions and attributions
        """
        return self._model.explain_batch(input_dicts, n_steps=n_steps)

    def save_vocabulary(self, directory: str) -> None:
        """Saves the pipeline's vocabulary in a directory

        Parameters
        ----------
        directory: str
        """
        self._model.vocab.save_to_files(directory)

    def serve(self, port: int = 9998):
        """Launches a REST prediction service with the current model

        Parameters
        ----------
        port: `int`
            The port on which the prediction service will be running (default: 9998)
        """
        from ._helpers import _serve

        self._model = self._model.eval()
        return _serve(self, port)

    def set_head(self, type: Type[TaskHead], **kwargs):
        """Sets a new task head for the pipeline

        Call this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.

        Parameters
        ----------
        type: `Type[TaskHead]`
            The `TaskHead` class to be set for the pipeline (e.g., `TextClassification`
        **kwargs:
            The `TaskHead` specific arguments (e.g., the classification head needs a `pooler` layer)
        """

        self._config.head = TaskHeadConfiguration(type=type, **kwargs)
        self._model.set_head(self._config.head.compile(backbone=self.backbone))

    @property
    def name(self) -> str:
        """Gets the pipeline name"""
        return self._model.name

    @property
    def inputs(self) -> List[str]:
        """Gets the pipeline input field names"""
        return self._model.inputs

    @property
    def output(self) -> List[str]:
        """Gets the pipeline output field names"""
        return self._model.output

    @property
    def backbone(self) -> ModelBackbone:
        """Gets the model backbone of the pipeline"""
        return self.head.backbone

    @property
    def head(self) -> TaskHead:
        """Gets the pipeline task head"""
        return self._model.head

    @property
    def config(self) -> PipelineConfiguration:
        """Gets the pipeline configuration"""
        return self._config

    @property
    def type_name(self) -> str:
        """The pipeline name. Equivalent to task head name"""
        return self.head.__class__.__name__

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters present in the model.

        At training time, this number can change when freezing/unfreezing certain parameter groups.
        """
        if vocabulary.is_empty(self._model.vocab, self.config.features.keys):
            self.__LOGGER.warning(
                "Your vocabulary is still empty! "
                "The number of trainable parameters usually depend on the size of your vocabulary."
            )
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def num_parameters(self) -> int:
        """Number of parameters present in the model."""
        if vocabulary.is_empty(self._model.vocab, self.config.features.keys):
            self.__LOGGER.warning(
                "Your vocabulary is still empty! "
                "The number of trainable parameters usually depend on the size of your vocabulary."
            )
        return sum(p.numel() for p in self._model.parameters())

    @property
    def named_trainable_parameters(self) -> List[str]:
        """Returns the names of the trainable parameters in the pipeline"""
        return [name for name, p in self._model.named_parameters() if p.requires_grad]

    def _update_prediction_signatures(self):
        """Fixes the `self.predict` signature to match the model inputs for interactive work-flows"""
        new_signature = inspect.Signature(
            [
                Parameter(name=_input, kind=Parameter.POSITIONAL_OR_KEYWORD)
                for _input in self.inputs
            ]
        )

        for method in [self.predict, self.explain]:
            self.__setattr__(
                method.__name__, update_method_signature(new_signature, method)
            )

    @staticmethod
    def _model_from_config(
        config: PipelineConfiguration,
        vocab: Optional[Vocabulary] = None,
        **extra_params,
    ) -> PipelineModel:
        """Creates a internal base model from a pipeline configuration

        Parameters
        ----------
        config
            Configuration of the pipeline
        vocab
            Vocabulary for the pipeline
        **extra_params

        Returns
        -------
        pipeline_model
        """
        return PipelineModel.from_params(
            Params({"config": config}), vocab=vocab, **extra_params
        )

    def _extend_vocabulary(
        self, vocab: Vocabulary, vocab_config: VocabularyConfiguration
    ) -> Vocabulary:
        """
        Extends a data vocabulary from a given configuration.

        The source vocabulary `vocab` won't be changed, instead a new vocabulary is created
        that includes the source vocabulary `vocab` and a vocabulary created from `vocab_config`

        Parameters
        ----------
        vocab
            The source vocabulary
        vocab_config
            The vocab extension configuration

        Returns
        -------
        extended_vocab
            An extended `Vocabulary` using the provided configuration
        """
        datasets = []
        for source in vocab_config.sources:
            if isinstance(source, Dataset):
                datasets.append(source.to_instances(pipeline=self))
            else:
                datasets.append(source)

        instances_vocab = Vocabulary.from_instances(
            instances=(instance for dataset in datasets for instance in dataset),
            max_vocab_size=vocab_config.max_vocab_size,
            min_count=vocab_config.min_count,
            pretrained_files=vocab_config.pretrained_files,
            only_include_pretrained_words=vocab_config.only_include_pretrained_words,
            min_pretrained_embeddings=vocab_config.min_pretrained_embeddings,
            tokens_to_add=vocab_config.tokens_to_add,
        )
        instances_vocab.extend_from_vocab(vocab)

        return instances_vocab

    def has_empty_vocab(self) -> bool:
        """Determines if a pipeline has an empty vocab under configured features"""
        return vocabulary.is_empty(self.backbone.vocab, self.config.features.keys)


class _BlankPipeline(Pipeline):
    """A blank pipeline initialized via a configuration

    Parameters
    ----------
    config: `Optional[PipelineConfiguration]`
        A `PipelineConfiguration` object defining the configuration of the fresh `Pipeline`.
    vocab
        The vocabulary for the pipeline
    """

    def __init__(
        self,
        config: PipelineConfiguration,
        vocab: Optional[Vocabulary] = None,
        **extra_args,
    ):
        self._config = config
        self._model = self._model_from_config(self._config, vocab=vocab, **extra_args)

        if not isinstance(self._model, PipelineModel):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")

        self._add_transformers_vocab_if_necessary(self._model.vocab)
        self._update_prediction_signatures()

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        """Creates the vocabulary for the pipeline from scratch

        Parameters
        ----------
        config
            Specifies the sources of the vocabulary and how to extract it
        """
        vocab = self._extend_vocabulary(vocabulary.create_empty_vocabulary(), config)
        # TODO (dcfidalgo): This can maybe optimized, do we really need to create a new PipelineModel
        #  and add again the transformers vocab?
        self._model = self._model_from_config(self.config, vocab=vocab)
        self._add_transformers_vocab_if_necessary(self._model.vocab)

    def _add_transformers_vocab_if_necessary(self, vocab):
        """Adds the transformers vocabulary to the `vocab`

        Parameters
        ----------
        vocab
            The transformers vocabulary will be added to this vocab
        """
        # The AllenNLP`s PretrainedTransformerIndexer adds its specific vocabulary to the Model's vocab
        # when the first `tokens_to_index()` is called via the private _add_encoding_to_vocabulary_if_needed method.
        # We trigger this here manually in a super ugly way ...
        # Actually i am not sure why they add it to their vocab in the first place ...
        transformers_indexer = self.backbone.featurizer.indexer.get(
            TransformersFeatures.namespace
        )
        if transformers_indexer is not None:
            try:
                transformers_indexer._add_encoding_to_vocabulary_if_needed(vocab)
            except AttributeError:
                transformers_indexer._matched_indexer._add_encoding_to_vocabulary_if_needed(
                    vocab
                )


class _PreTrainedPipeline(Pipeline):
    """

    Arguments
    ---------
    pretrained_path: `Optional[str]`
        The path to the model.tar.gz of a pre-trained `Pipeline`

    """

    def __init__(self, pretrained_path: str, **extra_args):
        self._binary = pretrained_path
        archive = load_archive(
            self._binary,
            # Necessary for AllenNLP>=1.2.0 that requires a dataset_reader config key
            # We choose the "interleaving" type since it is the most light weight one.
            overrides={"dataset_reader": {"type": "interleaving", "readers": {}}},
            **extra_args,
        )
        self._model = self.__model_from_archive(archive)
        self._config = self.__config_from_archive(archive)

        if not isinstance(self._model, PipelineModel):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")
        self._update_prediction_signatures()

    @staticmethod
    def __model_from_archive(archive: Archive) -> PipelineModel:
        if not isinstance(archive.model, PipelineModel):
            raise ValueError(f"Wrong pipeline model: {archive.model}")
        return cast(PipelineModel, archive.model)

    @staticmethod
    def __config_from_archive(archive: Archive) -> PipelineConfiguration:
        config = archive.config["model"]["config"]
        return PipelineConfiguration.from_params(config)

    @property
    def trained_path(self) -> str:
        """Gets the path to the pretrained binary file"""
        return self._binary


class _PipelineCopy(Pipeline):
    """A copy of a pipeline ready for training."""

    def __init__(self, from_pipeline: Pipeline):
        self._model = self._model_from_config(
            from_pipeline.config, vocab=from_pipeline.backbone.vocab
        )
        if isinstance(from_pipeline, _PreTrainedPipeline):
            self._model.load_state_dict(from_pipeline._model.state_dict())

        self._config = copy.deepcopy(from_pipeline.config)
