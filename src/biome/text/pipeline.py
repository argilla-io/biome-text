import copy
import inspect
import logging
import os
import shutil
import uuid
from inspect import Parameter
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast

import numpy
from allennlp.common import Params
from allennlp.data import AllennlpDataset, AllennlpLazyDataset, Instance, Vocabulary
from allennlp.models import load_archive
from allennlp.models.archival import Archive
from dask import dataframe as dd
from dask.dataframe import DataFrame

from biome.text import vocabulary
from biome.text.configuration import (
    PipelineConfiguration,
    TrainerConfiguration,
    VocabularyConfiguration,
)
from biome.text.data import DataSource, InstancesDataset
from biome.text.errors import ActionNotSupportedError, EmptyVocabError
from biome.text.helpers import update_method_signature
from . import constants
from ._configuration import ElasticsearchExplore, ExploreConfiguration
from ._model import PipelineModel
from .backbone import ModelBackbone
from .modules.heads import TaskHead, TaskHeadConfiguration
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
    __TRAINING_CACHE_DATA = ".instances_data"
    __DATASOURCE_YAML_FOLDER = ".datasources"

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
    ) -> "Pipeline":
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
    def from_pretrained(cls, path: str, **kwargs) -> "Pipeline":
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

    def train(
        self,
        output: str,
        training: Union[DataSource, InstancesDataset],
        trainer: Optional[TrainerConfiguration] = None,
        validation: Optional[Union[DataSource, InstancesDataset]] = None,
        test: Optional[Union[DataSource, InstancesDataset]] = None,
        extend_vocab: Optional[VocabularyConfiguration] = None,
        epoch_callbacks: List["allennlp.training.EpochCallback"] = None,
        restore: bool = False,
        quiet: bool = False,
    ) -> TrainingResults:
        """Launches a training run with the specified configurations and data sources

        Parameters
        ----------
        output:
            The experiment output path
        training:
            The training DataSource
        trainer:
            The trainer file path
        validation:
            The validation DataSource (optional)
        test:
            The test DataSource (optional)
        extend_vocab:
            Extends the vocabulary tokens with the provided VocabularyConfiguration
        epoch_callbacks:
            A list of callbacks that will be called at the end of every epoch, and at the start of
            training (with epoch = -1).
        restore:
            If enabled, tries to read previous training status from the `output` folder and
            continues the training process
        quiet:
            If enabled, disables most logging messages keeping only warning and error messages.
            In any case, all logging info will be stored into a file at ${output}/train.log

        Returns
        -------

        Training results information, containing the generated model path and the related metrics
        """
        if extend_vocab is not None and isinstance(self, _BlankPipeline):
            raise ActionNotSupportedError(
                "If you want to customize pipeline vocab, please use create_vocab method instead"
            )

        trainer = trainer or TrainerConfiguration()
        try:
            if not restore and os.path.isdir(output):
                shutil.rmtree(output)

            self.__configure_training_logging(output, quiet)

            # The original pipeline keeps unchanged
            train_pipeline = self.__make_copy()
            vocab = None

            if restore:
                vocab = vocabulary.load_vocabulary(os.path.join(output, "vocabulary"))
            if extend_vocab is not None and not vocab:
                vocab = train_pipeline._extend_vocabulary(
                    train_pipeline.backbone.vocab, vocab_config=extend_vocab
                )
            if vocab:
                train_pipeline._set_vocab(vocab)

            if vocabulary.is_empty(
                train_pipeline.backbone.vocab, self.config.features.keys
            ):
                raise EmptyVocabError(
                    "Found an empty vocabulary. "
                    "You probably forgot to create a vocabulary with '.create_vocabulary()'."
                )

            from ._helpers import PipelineTrainer

            datasets = {"training": training, "validation": validation, "test": test}
            for name, dataset in datasets.items():
                if isinstance(dataset, DataSource):
                    datasets[name] = train_pipeline.create_dataset(dataset)

            trainer = PipelineTrainer(
                train_pipeline,
                trainer_config=trainer,
                output_dir=output,
                epoch_callbacks=epoch_callbacks,
                **datasets,
            )

            model_path, metrics = trainer.train()
            return TrainingResults(model_path, metrics)

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

    def __make_copy(self) -> "Pipeline":
        """
        Creates a copy of current pipeline instance
        """
        if isinstance(self, _BlankPipeline):
            return _BlankPipeline(config=self.config, vocab=self.backbone.vocab)
        if isinstance(self, _PreTrainedPipeline):
            return Pipeline.from_pretrained(self.trained_path)
        raise ValueError(f"Cannot clone pipeline {self}")

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

    def create_dataset(
        self, datasource: DataSource, lazy: bool = False
    ) -> InstancesDataset:
        """
        Creates an instances torch Dataset from an data source

        Parameters
        ----------
        datasource:
            The source of data
        lazy:
            If enabled, the returned dataset is a subclass of `torch.data.utils.IterableDataset`

        Returns
        -------

        A torch Dataset containing the instances collection

        """
        mapping = {k: k for k in self.inputs + [self.output] if k}
        mapping.update(datasource.mapping)

        datasource.mapping = mapping
        ddf = datasource.to_mapped_dataframe()
        instances_ddf = ddf.map_partitions(
            lambda df: df.apply(
                lambda row: self.head.featurize(**row.to_dict()), axis=1
            ),
            meta=object,
        ).persist()

        def build_instance_generator(instances: DataFrame):
            """Configures an instance generator from DataFrame"""

            def instance_generator(path: str) -> Iterable[Instance]:
                yield from instances

            return instance_generator

        return (
            AllennlpLazyDataset(
                instance_generator=build_instance_generator(instances_ddf),
                file_path="dummy",
            )
            if lazy
            else AllennlpDataset(instances_ddf.compute())
        )

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

    def save_vocabulary(self, directory: str) -> None:
        """Saves the pipeline's vocabulary in a directory

        Parameters
        ----------
        directory: str
        """
        self._model.vocab.save_to_files(directory)

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        """Creates a vocabulary an set it to pipeline"""
        raise NotImplementedError

    def explore(
        self,
        data_source: DataSource,
        explore_id: Optional[str] = None,
        es_host: Optional[str] = None,
        batch_size: int = 50,
        prediction_cache_size: int = 0,
        explain: bool = False,
        force_delete: bool = True,
        **metadata,
    ) -> dd.DataFrame:
        """Launches the Explore UI for a given data source

        Running this method inside an `IPython` notebook will try to render the UI directly in the notebook.

        Running this outside a notebook will try to launch the standalone web application.

        Parameters
        ----------
        data_source: `DataSource`
            The data source or its yaml file path
        explore_id: `Optional[str]`
            A name or id for this explore run, useful for running and keep track of several explorations
        es_host: `Optional[str]`
            The URL to the Elasticsearch host for indexing predictions (default is `localhost:9200`)
        batch_size: `int`
            The batch size for indexing predictions (default is `500)
        prediction_cache_size: `int`
            The size of the cache for caching predictions (default is `0)
        explain: `bool`
            Whether to extract and return explanations of token importance (default is `False`)
        force_delete: `bool`
            Deletes exploration with the same `explore_id` before indexing the new explore items (default is `True)

        Returns
        -------
        pipeline: `Pipeline`
            A configured pipeline
        """
        from ._helpers import _explore, _show_explore

        config = ExploreConfiguration(
            batch_size=batch_size,
            prediction_cache_size=prediction_cache_size,
            explain=explain,
            force_delete=force_delete,
            **metadata,
        )

        es_config = ElasticsearchExplore(
            es_index=explore_id or str(uuid.uuid1()),
            es_host=es_host or constants.DEFAULT_ES_HOST,
        )

        if not data_source.mapping:
            data_source.mapping = self._model._default_ds_mapping
        explore_df = _explore(self, data_source, config, es_config)
        _show_explore(es_config)

        return explore_df

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

        self._config.head = TaskHeadConfiguration(type=type.__name__, **kwargs)
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
    def output(self) -> str:
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
    def trainable_parameters(self) -> int:
        """
        Returns the number of trainable parameters.

        At training time, this number can change when freezing/unfreezing certain parameter groups.
        """
        if vocabulary.is_empty(self._model.vocab, self.config.features.keys):
            self.__LOGGER.warning(
                "Your vocabulary is still empty! "
                "The number of trainable parameters usually depend on the size of your vocabulary."
            )
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def trainable_parameter_names(self) -> List[str]:
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

    def _extend_vocabulary(
        self, vocab: Vocabulary, vocab_config: VocabularyConfiguration
    ) -> Vocabulary:
        """
        Extends a data vocabulary from a given configuration.

        The source vocabulary won't be changed, instead of that, a new vocabulary will be created
        including source vocab with extended configuration

        Parameters
        ----------
        vocab: `Vocabulary`
            The source vocabulary
        vocab_config: `VocabularyConfiguration`
            The vocab extension configuration

        Returns
        -------
        vocab: `Vocabulary`
            An extended `Vocabulary` using the provided configuration
        """

        datasets = [
            self.create_dataset(source) if isinstance(source, DataSource) else source
            for source in vocab_config.sources
        ]

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


class _BlankPipeline(Pipeline):
    """
    Parameters
    ----------
    config: `Optional[PipelineConfiguration]`
        A `PipelineConfiguration` object defining the configuration of the fresh `Pipeline`.
    """

    def __init__(self, config: PipelineConfiguration, **extra_args):
        self._config = config
        self._model = self.__model_from_config(self._config, **extra_args)
        if not isinstance(self._model, PipelineModel):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")
        self._update_prediction_signatures()

    @staticmethod
    def __model_from_config(
        config: PipelineConfiguration, **extra_params
    ) -> PipelineModel:
        """Creates a internal base model from a pipeline configuration"""
        return PipelineModel.from_params(Params({"config": config}), **extra_params)

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        vocab = self._extend_vocabulary(Vocabulary(), config)
        self._model = self.__model_from_config(self.config, vocab=vocab)


class _PreTrainedPipeline(Pipeline):
    """

    Arguments
    ---------
    pretrained_path: `Optional[str]`
        The path to the model.tar.gz of a pre-trained `Pipeline`

    """

    def __init__(self, pretrained_path: str, **extra_args):
        self._binary = pretrained_path
        archive = load_archive(self._binary, **extra_args)
        self._model = self.__model_from_archive(archive)
        self._config = self.__config_from_archive(archive)

        if not isinstance(self._model, PipelineModel):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")
        self._update_prediction_signatures()

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        raise ActionNotSupportedError(
            "Cannot create a vocabulary for an already pretrained model!!!"
        )

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
