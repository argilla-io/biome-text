import copy
import glob
import inspect
import logging
import os
import shutil
import tempfile
import uuid
from inspect import Parameter
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models import load_archive
from allennlp.models.archival import Archive
from dask import dataframe as dd

from biome.text import vocabulary
from biome.text.configuration import (
    PipelineConfiguration,
    TrainerConfiguration,
    VocabularyConfiguration,
)
from biome.text.data import DataSource
from biome.text.errors import ActionNotSupportedError, EmptyVocabError
from biome.text.helpers import update_method_signature
from . import constants
from ._configuration import (
    ElasticsearchExplore,
    ExploreConfiguration,
    TrainConfiguration,
    _ModelImpl,
)
from .backbone import ModelBackbone
from .modules.heads import TaskHead, TaskHeadSpec

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

    _model: _ModelImpl = None
    _config: PipelineConfiguration = None

    @classmethod
    def from_yaml(cls, path: str, vocab_path: Optional[str] = None) -> "Pipeline":
        """Creates a pipeline from a config yaml file path

        Parameters
        ----------
        path: `str`
            The path to a YAML configuration file
        vocab_path: `Optional[str]`
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
        """Creates a pipeline from a `PipelineConfiguration` object

        Parameters
        ----------
        config: `Union[PipelineConfiguration, dict]`
            A `PipelineConfiguration` object or a configuration dict
        vocab_path: `Optional[str]`
            If provided, the pipeline vocab will be loaded from this path

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
        """Loads a pipeline from a pre-trained pipeline from a model.tar.gz file path

        Parameters
        ----------
            path: `str`
                The path to the model.tar.gz file of a pre-trained `Pipeline`

        Returns
        -------
            pipeline: `Pipeline`
                A configured pipeline
        """
        return _PreTrainedPipeline(pretrained_path=path, **kwargs)

    def init_prediction_logger(self, output_dir: str, max_logging_size: int = 100):
        """Initialize the prediction logging.

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
        """Initialize cache for input predictions

        Parameters
        ----------
        max_size
            Save up to max_size most recent (inputs).
        """
        self._model.init_prediction_cache(max_size)

    def train(
        self,
        output: str,
        training: DataSource,
        trainer: Optional[TrainerConfiguration] = None,
        validation: Optional[DataSource] = None,
        test: Optional[DataSource] = None,
        extend_vocab: Optional[VocabularyConfiguration] = None,
        restore: bool = False,
    ) -> None:
        """Launches a training run with the specified configurations and data sources

        Parameters
        ----------
        output: `str`
            The experiment output path
        training: `DataSource`
            The training data source
        trainer: `TrainerConfiguration`
            The trainer file path
        validation: `Optional[DataSource]`
            The validation data source
        test: `Optional[DataSource]`
            The test data source
        extend_vocab: `Optional[VocabularyConfiguration]`
            Extends vocab tokens with provided configuration
        restore: `bool`
            If enabled, tries to read previous training status from output folder and
            continues training process from it
        """
        trainer = trainer or TrainerConfiguration()

        from ._helpers import _allennlp_configuration

        allennlp_logger = logging.getLogger("allennlp")

        try:
            allennlp_logger.setLevel(logging.INFO)

            if not restore and os.path.isdir(output):
                shutil.rmtree(output)

            # The original pipeline keeps unchanged
            model = copy.deepcopy(self._model)
            # creates the output folder if it does not exist
            model.cache_data(os.path.join(output, self.__TRAINING_CACHE_DATA))

            vocab = None
            if restore:
                vocab = vocabulary.load_vocabulary(os.path.join(output, "vocabulary"))
            if extend_vocab and not vocab:
                vocab = self._extend_vocabulary(model.vocab, vocab_config=extend_vocab)
            if vocab:
                model.set_vocab(vocab)

            if vocabulary.is_empty(model.vocab, self.config.features.keys):
                raise EmptyVocabError(
                    "Found an empty vocabulary. "
                    "You probably forgot to create a vocabulary with '.create_vocabulary()'."
                )

            # `_allennlp_configuration` needs strings
            datasources_dir = os.path.join(output, self.__DATASOURCE_YAML_FOLDER)
            training = training.to_yaml(
                os.path.join(
                    datasources_dir, f"training_{os.path.basename(training.source)}.yml"
                ),
                make_source_path_absolute=True,
            )
            if validation is not None:
                validation = validation.to_yaml(
                    os.path.join(
                        datasources_dir,
                        f"validation_{os.path.basename(validation.source)}.yml",
                    ),
                    make_source_path_absolute=True,
                )
            if test is not None:
                test = test.to_yaml(
                    os.path.join(
                        datasources_dir, f"test_{os.path.basename(test.source)}.yml"
                    ),
                    make_source_path_absolute=True,
                )

            config = TrainConfiguration(
                output=output,
                trainer=trainer,
                training=training,
                validation=validation,
                test=test,
            )

            model.launch_experiment(
                params=Params(_allennlp_configuration(self, config)),
                serialization_dir=output,
            )
        finally:
            allennlp_logger.setLevel(logging.ERROR)

    def predict(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        """Predicts over some input data with current state of the model

        The accepted input is dynamically calculated from head.input

        Returns
        -------
        predictions: `Dict[str, numpy.ndarray]`
            A dictionary containing the predictions and additional information
        """
        return self._model.predict(*args, **kwargs)

    def explain(self, *args, n_steps: int = 5, **kwargs) -> Dict[str, Any]:
        """Predicts over some input data with current state of the model and provides explanations of token importance.

        The accepted input is dynamically calculated from head.input

        Parameters
        ----------

        n_steps: int
            The number of steps for token attribution calculation (if proceed).
            If the number of steps is less than 1, the attributions will not be calculated

        Returns
        -------
        predictions: `Dict[str, numpy.ndarray]`
            A dictionary containing the predictions with token importance calculated using IntegratedGradients
        """
        return self._model.explain(*args, n_steps=n_steps, **kwargs)

    def save_vocabulary(self, path: str) -> None:
        """Save the pipeline vocabulary into a path"""
        self._model.vocab.save_to_files(path)

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        """Creates a vocabulary an set it to pipeline"""
        raise NotImplementedError

    def explore(
        self,
        data_source: DataSource,
        explore_id: Optional[str] = None,
        es_host: Optional[str] = None,
        batch_size: int = 500,
        prediction_cache_size: int = 0,
        explain: bool = False,
        force_delete: bool = True,
        **metadata,
    ) -> dd.DataFrame:
        """Launches Explore UI for a given data source with current model

        Running this method inside a an `IPython` notebook will try to render the UI directly in the notebook.

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
        from ._helpers import (
            _explore,
            _show_explore,
        )

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

        explore_df = _explore(self, data_source, config, es_config)
        _show_explore(es_config)

        return explore_df

    def serve(self, port: int = 9998):
        """Launches a REST prediction service with current model in a specified port (default is `9998)

        # Parameters
            port: `int`
                The port to make available the prediction service
        """
        from ._helpers import _serve

        self._model = self._model.eval()
        return _serve(self, port)

    def set_head(self, type: Type[TaskHead], **params):
        """Sets a new task head for the pipeline

        Use this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.

        Parameters
        ----------
        type: `Type[TaskHead]`
            The `TaskHead` class to be set for the pipeline (e.g., `TextClassification`
        params: `**kwargs`
            The `TaskHead` specific parameters (e.g., classification head needs a `pooler` layer)
        """

        self._config.head = TaskHeadSpec(type=type.__name__, **params)
        self._model.set_head(self._config.head.compile(backbone=self.backbone))

    @property
    def name(self):
        """Gets pipeline  name"""
        return self._model.name

    @property
    def inputs(self) -> List[str]:
        """Gets pipeline input field names"""
        return self._model.inputs

    @property
    def output(self) -> str:
        """Gets pipeline output field names"""
        return self._model.output

    @property
    def backbone(self) -> ModelBackbone:
        """Gets pipeline backbone encoder"""
        return self.head.backbone

    @property
    def head(self) -> TaskHead:
        """Gets pipeline task head"""
        return self._model.head

    @property
    def config(self) -> PipelineConfiguration:
        return self._config

    @property
    def type_name(self) -> str:
        """The pipeline name. Equivalent to task head name"""
        return self.head.__class__.__name__

    @property
    def trainable_parameters(self) -> int:
        """
        Return the number of trainable parameters.

        This number could be change before an after a training process, since trainer could fix some of them.

        """
        if vocabulary.is_empty(self._model.vocab, self.config.features.keys):
            self.__LOGGER.warning("Your vocabulary is still empty! "
                                  "The number of trainable parameters usually depend on the size of your vocabulary.")
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def trainable_parameter_names(self) -> List[str]:
        """Returns the name of pipeline trainable parameters"""
        return [name for name, p in self._model.named_parameters() if p.requires_grad]

    def _update_prediction_signatures(self):
        """For interactive work-flows, fixes the predict signature to the model inputs"""
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
        Extends a data vocabulary from a given configuration

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
        source_paths = [
            source.to_yaml(
                tempfile.NamedTemporaryFile(delete=False).name,
                make_source_path_absolute=True,
            )
            for source in vocab_config.sources
        ]
        instances_vocab = Vocabulary.from_instances(
            instances=(
                instance
                for data_source in source_paths
                for instance in self._model.read(data_source)
            ),
            max_vocab_size=vocab_config.max_vocab_size,
            min_count=vocab_config.min_count,
            pretrained_files=vocab_config.pretrained_files,
            only_include_pretrained_words=vocab_config.only_include_pretrained_words,
            min_pretrained_embeddings=vocab_config.min_pretrained_embeddings,
            tokens_to_add=vocab_config.tokens_to_add,
        )
        vocab.extend_from_vocab(instances_vocab)
        return vocab


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
        if not isinstance(self._model, _ModelImpl):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")
        self._update_prediction_signatures()

    @staticmethod
    def __model_from_config(
        config: PipelineConfiguration, **extra_params
    ) -> _ModelImpl:
        """Creates a internal base model from pipeline configuration"""
        return _ModelImpl.from_params(Params({"config": config}), **extra_params)

    def train(
        self,
        output: str,
        training: DataSource,
        trainer: Optional[TrainerConfiguration] = None,
        validation: Optional[DataSource] = None,
        test: Optional[DataSource] = None,
        extend_vocab: Optional[VocabularyConfiguration] = None,
        restore: bool = False,
    ) -> None:
        if extend_vocab:
            raise ActionNotSupportedError(
                "If you want to customize pipeline vocab, please use create_vocab method instead"
            )
        super(_BlankPipeline, self).train(
            output=output,
            training=training,
            trainer=trainer,
            validation=validation,
            test=test,
            restore=restore,
        )

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        vocab = self._extend_vocabulary(Vocabulary(), config)
        # TODO: on Allennlp 1.0 we can use model.set_vocab.
        #  For now, we must reload the model passing vocab to allow restore vocab in train
        # self._model.set_vocab(vocab)
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

        if not isinstance(self._model, _ModelImpl):
            raise TypeError(f"Cannot load model. Wrong format of {self._model}")
        self._update_prediction_signatures()

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        raise ActionNotSupportedError(
            "Cannot create a vocabulary for an already pretrained model!!!"
        )

    @staticmethod
    def __model_from_archive(archive: Archive) -> _ModelImpl:
        if not isinstance(archive.model, _ModelImpl):
            raise ValueError(f"Wrong pipeline model: {archive.model}")
        return cast(_ModelImpl, archive.model)

    @staticmethod
    def __config_from_archive(archive: Archive) -> PipelineConfiguration:
        config = archive.config["model"]["config"]
        return PipelineConfiguration.from_params(config)

    @property
    def trained_path(self) -> str:
        """Path to binary file when load from binary"""
        return self._binary
