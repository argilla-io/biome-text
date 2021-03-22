import copy
import inspect
import json
import logging
import os
import shutil
import tempfile
from inspect import Parameter
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import mlflow
import numpy
import torch
from allennlp.commands.find_learning_rate import search_learning_rate
from allennlp.common import Params
from allennlp.common.file_utils import is_url_or_existing_file
from allennlp.data import AllennlpLazyDataset
from allennlp.data import Vocabulary
from allennlp.models import load_archive
from allennlp.models.archival import Archive
from allennlp.models.archival import archive_model
from allennlp.training.util import evaluate

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.configuration import FindLRConfiguration
from biome.text.configuration import PipelineConfiguration
from biome.text.configuration import PredictionConfiguration
from biome.text.configuration import TrainerConfiguration
from biome.text.configuration import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.dataset import InstanceDataset
from biome.text.errors import EmptyVocabError
from biome.text.features import TransformersFeatures
from biome.text.features import WordFeatures
from biome.text.helpers import update_method_signature
from biome.text.loggers import BaseTrainLogger
from biome.text.loggers import add_default_wandb_logger_if_needed
from biome.text.model import PipelineModel
from biome.text.modules.heads import TaskHead
from biome.text.modules.heads import TaskHeadConfiguration
from biome.text.training_results import TrainingResults

logging.getLogger("allennlp").setLevel(logging.ERROR)
logging.getLogger("elasticsearch").setLevel(logging.ERROR)


class Pipeline:
    """Manages NLP models configuration and actions.

    Use `Pipeline` for creating new models from a configuration or loading a pretrained model.

    Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.
    """

    _LOGGER = logging.getLogger(__name__)

    def __init__(self, model: PipelineModel, config: PipelineConfiguration):
        self._model = model
        self._config = config

        self._update_prediction_signatures()

    def _update_prediction_signatures(self):
        """Updates the `self.predict` signature to match the model inputs for interactive work-flows"""
        updated_parameters = [
            par
            for name, par in inspect.signature(self.head.featurize).parameters.items()
            if par.default == Parameter.empty
        ] + [
            par
            for name, par in inspect.signature(self.predict).parameters.items()
            if name not in ["args", "kwargs"]
        ]
        new_signature = inspect.Signature(updated_parameters)

        self.__setattr__(
            self.predict.__name__, update_method_signature(new_signature, self.predict)
        )

    @classmethod
    def from_yaml(cls, path: str, vocab_path: Optional[str] = None) -> "Pipeline":
        """Creates a pipeline from a config yaml file

        Parameters
        ----------
        path
            The path to a YAML configuration file
        vocab_path
            If provided, the pipeline vocab will be loaded from this path

        Returns
        -------
        pipeline
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
        config
            A `PipelineConfiguration` object or a configuration dict
        vocab_path
            If provided, the pipeline vocabulary will be loaded from this path

        Returns
        -------
        pipeline
            A configured pipeline
        """
        if isinstance(config, dict):
            config = PipelineConfiguration.from_dict(config)

        model = PipelineModel.from_params(
            Params({"config": config}),
            vocab=Vocabulary.from_files(vocab_path) if vocab_path is not None else None,
        )
        if not isinstance(model, PipelineModel):
            raise TypeError(f"Cannot load model. Wrong format of {model}")

        cls._add_transformers_vocab_if_needed(model)

        return cls(model, config)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "Pipeline":
        """Loads a pretrained pipeline providing a *model.tar.gz* file path

        Parameters
        ----------
        path
            The path to the *model.tar.gz* file of a pretrained `Pipeline`

        Returns
        -------
        pipeline
            A pretrained pipeline
        """
        archive = load_archive(
            path,
            # Necessary for AllenNLP>=1.2.0 that requires a dataset_reader config key
            # We choose the "interleaving" type since it is the most light weight one.
            overrides={"dataset_reader": {"type": "interleaving", "readers": {}}},
        )
        model = cls._model_from_archive(archive)
        model.file_path = str(path)
        config = cls._config_from_archive(archive)

        if not isinstance(model, PipelineModel):
            raise TypeError(f"Cannot load model. Wrong format of {model}")

        return cls(model, config)

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
    def vocab(self) -> Vocabulary:
        """Gets the pipeline vocabulary"""
        return self._model.vocab

    @property
    def config(self) -> PipelineConfiguration:
        """Gets the pipeline configuration"""
        return self._config

    @property
    def model(self) -> PipelineModel:
        """Gets the underlying model"""
        return self._model

    @property
    def type_name(self) -> str:
        """The pipeline name. Equivalent to task head name"""
        return self.head.__class__.__name__

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters present in the model.

        At training time, this number can change when freezing/unfreezing certain parameter groups.
        """
        if vocabulary.is_empty(self.vocab, self.config.features.configured_namespaces):
            self._LOGGER.warning(
                "At least one vocabulary of your features is still empty! "
                "The number of trainable parameters usually depends on the size of your vocabulary."
            )
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def num_parameters(self) -> int:
        """Number of parameters present in the model."""
        if vocabulary.is_empty(self.vocab, self.config.features.configured_namespaces):
            self._LOGGER.warning(
                "At least one vocabulary of your features is still empty! "
                "The number of trainable parameters usually depends on the size of your vocabulary."
            )
        return sum(p.numel() for p in self._model.parameters())

    @property
    def named_trainable_parameters(self) -> List[str]:
        """Returns the names of the trainable parameters in the pipeline"""
        return [name for name, p in self._model.named_parameters() if p.requires_grad]

    @property
    def model_path(self) -> str:
        """Returns the file path to the serialized version of the last trained model"""
        return self._model.file_path

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
        training_data: Dataset,
        vocab_config: Optional[Union[VocabularyConfiguration, str]] = "default",
        lazy: bool = False,
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
        vocab_config
            A `VocabularyConfiguration` to create/extend the pipeline's vocabulary if necessary.
            If 'default' (str), we will use the default configuration
            `VocabularyConfiguration(datasets=[training_data])`.
            If None, we will leave the pipeline's vocabulary untouched.
        lazy
            If true, dataset instances are lazily loaded from disk, otherwise they are loaded and kept in memory.

        Returns
        -------
        (learning_rates, losses)
            Returns a list of learning rates and corresponding losses.
            Note: The losses are recorded before applying the corresponding learning rate
        """
        from biome.text._helpers import create_trainer_for_finding_lr

        self._prepare_vocab(
            vocab_config=vocab_config, training_data=training_data, lazy=lazy
        )

        if isinstance(training_data, Dataset):
            training_data: AllennlpLazyDataset = training_data.to_instances(
                pipeline=self,
                lazy=lazy,
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
        training: Dataset,
        trainer: Optional[TrainerConfiguration] = None,
        validation: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        vocab_config: Optional[Union[VocabularyConfiguration, str]] = "default",
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
        vocab_config
            A `VocabularyConfiguration` to create/extend the pipeline's vocabulary if necessary.
            If 'default' (str), we will use the default configuration `VocabularyConfiguration(datasets=[training])`.
            If None, we will leave the pipeline's vocabulary untouched.
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
        from ._helpers import PipelineTrainer

        trainer = trainer or TrainerConfiguration()

        if not restore and os.path.isdir(output):
            shutil.rmtree(output)

        try:
            self.__configure_training_logging(output, quiet)

            self._prepare_vocab(
                vocabulary_folder=os.path.join(output, "vocabulary")
                if restore
                else None,
                vocab_config=vocab_config,
                training_data=training,
                lazy=lazy,
            )

            datasets = {"training": training, "validation": validation, "test": test}
            for name, dataset in datasets.items():
                if isinstance(dataset, Dataset):
                    datasets[name] = dataset.to_instances(pipeline=self, lazy=lazy)

            loggers = loggers or []
            loggers = add_default_wandb_logger_if_needed(loggers)

            pipeline_trainer = PipelineTrainer(
                self,
                trainer_config=trainer,
                output_dir=output,
                epoch_callbacks=loggers,
                **datasets,
            )

            training_results = pipeline_trainer.train()
            self._model.file_path = training_results.model_path

            return training_results

        finally:
            self.__restore_training_logging()

    def create_vocab(
        self,
        vocab_config: VocabularyConfiguration,
        lazy: bool = False,
    ):
        """Creates the vocab for the pipeline.

        NOTE: The trainer calls this method for you. You can use this method in case you want
        to create the vocab outside of the training/test process.

        Parameters
        ----------
        vocab_config
            Configurations for the vocab creation
        lazy
            If true, instances are lazily loaded from disk, otherwise they are loaded into memory.
        """
        # The transformers feature comes with its own vocab, no need to prepare anything if it is the only feature
        if self.config.features.configured_namespaces == [
            TransformersFeatures.namespace
        ]:
            return

        # If the vocab is empty, we assume this is an untrained pipeline
        # and we want to raise an error if the weights file is not found.
        # Extending the vocab with a non-existent weights file only throws a warning.
        try:
            assert is_url_or_existing_file(Path(self.config.features.word.weights_file))
        except AssertionError:
            if vocabulary.is_empty(self.vocab, [WordFeatures.namespace]):
                raise FileNotFoundError(
                    f"Cannot find the weights file {self.config.features.word.weights_file}"
                )
        # no word feature, or weights_file is None
        except (AttributeError, TypeError):
            pass

        vocab = vocab_config.build_vocab(pipeline=self, lazy=lazy)
        # If the vocab is the same, this is just a no-op
        self._model.extend_vocabulary(vocab)

    def _prepare_vocab(
        self,
        vocabulary_folder: Optional[str] = None,
        vocab_config: Optional[Union[str, VocabularyConfiguration]] = "default",
        training_data: Optional[Dataset] = None,
        lazy: bool = False,
    ):
        """Prepare and set the vocab for a training or learning rate scan.

        Parameters
        ----------
        vocabulary_folder
            If specified, load the vocab from this folder
        vocab_config
            A `VocabularyConfiguration` to create/extend the pipeline's vocabulary if necessary.
            If 'default' (str), we will use the default configuration
            `VocabularyConfiguration(datasets=[training_data])`.
            If None, we will leave the pipeline's vocabulary untouched.
        training_data
            The training data in case we need to construct the default config
        lazy
            If true, dataset instances are lazily loaded from disk, otherwise they are loaded and kept in memory.
        """
        # The transformers feature comes with its own vocab, no need to prepare anything if it is the only feature
        if self.config.features.configured_namespaces == [
            TransformersFeatures.namespace
        ]:
            return

        # If the vocab is empty, we assume this is an untrained pipeline
        # and we want to raise an error if the weights file is not found.
        # Extending the vocab with a non-existent weights file only throws a warning.
        try:
            assert is_url_or_existing_file(Path(self.config.features.word.weights_file))
        except AssertionError:
            if vocabulary.is_empty(self.vocab, [WordFeatures.namespace]):
                raise FileNotFoundError(
                    f"Cannot find the weights file {self.config.features.word.weights_file}"
                )
        # no word feature, or weights_file is None
        except (AttributeError, TypeError):
            pass

        if vocabulary_folder is not None:
            self._model.extend_vocabulary(Vocabulary.from_files(vocabulary_folder))
            vocab_config = None

        vocab_config = (
            VocabularyConfiguration(datasets=[training_data])
            if vocab_config == "default"
            else vocab_config
        )
        if vocab_config is not None:
            vocab = vocab_config.build_vocab(pipeline=self, lazy=lazy)
            self._model.extend_vocabulary(vocab)

        if vocabulary.is_empty(self.vocab, self.config.features.configured_namespaces):
            raise EmptyVocabError(
                "All your features need a non-empty vocabulary for a training!"
            )

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

    def predict(
        self,
        *args,
        batch: Optional[List[Dict[str, Any]]] = None,
        add_tokens: bool = False,
        add_attributions: bool = False,
        attributions_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Union[Dict[str, numpy.ndarray], List[Optional[Dict[str, numpy.ndarray]]]]:
        """Returns a prediction given some input data based on the current state of the model

        The accepted input is dynamically calculated and can be checked via the `self.inputs` attribute
        (`print(Pipeline.inputs)`)

        Parameters
        ----------
        *args/**kwargs
            These are dynamically updated and correspond to the pipeline's `self.inputs`.
            If provided, the `batch` parameter will be ignored.
        batch
            A list of dictionaries that represents a batch of inputs. The dictionary keys must comply with the
            `self.inputs` attribute. Predicting batches should typically be faster than repeated calls with args/kwargs.
        add_tokens
            If true, adds a 'tokens' key in the prediction that contains the tokenized input.
        add_attributions
            If true, adds a 'attributions' key that contains attributions of the input to the prediction.
        attributions_kwargs
            This dict is directly passed on to the `TaskHead.compute_attributions()`.

        Returns
        -------
        predictions
            A dictionary or a list of dictionaries containing the predictions and additional information.
            If a prediction fails for a single input in the batch, its return value will be `None`.

        Raises
        ------
        PredictionError
            Failed to predict the single input or the whole batch
        """
        if args or kwargs:
            batch = [self._map_args_kwargs_to_input(*args, **kwargs)]

        prediction_config = PredictionConfiguration(
            add_tokens=add_tokens,
            add_attributions=add_attributions,
            attributions_kwargs=attributions_kwargs or {},
        )

        predictions = self._model.predict(batch, prediction_config)
        if not any(predictions):
            raise PredictionError(f"Failed to make predictions for {batch}")

        predictions_dict = [
            prediction.as_dict() if prediction is not None else None
            for prediction in predictions
        ]

        return predictions_dict[0] if (args or kwargs) else predictions_dict

    def _map_args_kwargs_to_input(self, *args, **kwargs) -> Dict[str, Any]:
        """Helper function for the `self.predict` method"""
        input_dict = {k: v for k, v in zip(self.inputs, args)}
        input_dict.update(kwargs)

        return input_dict

    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        lazy: bool = False,
        cuda_device: int = None,
        predictions_output_file: Optional[str] = None,
        metrics_output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates the pipeline on a given dataset

        Parameters
        ----------
        dataset
            The dataset to use for the evaluation
        batch_size
            Batch size used during the evaluation
        lazy
            If true, instances from the dataset are lazily loaded from disk, otherwise they are loaded into memory.
        cuda_device
            If you want to use a specific CUDA device for the evaluation, specify it here. Pass on -1 for the CPU.
            By default we will use a CUDA device if one is available.
        predictions_output_file
            Optional path to write the predictions to.
        metrics_output_file
            Optional path to write the final metrics to.

        Returns
        -------
        metrics
            Metrics defined in the TaskHead
        """
        from biome.text._helpers import create_dataloader

        # move model to cuda device
        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1
        prior_device = next(self._model.parameters()).get_device()
        self._model.to(cuda_device if cuda_device >= 0 else "cpu")

        if not any(
            label_column in dataset.column_names for label_column in self.output
        ):
            raise ValueError(
                f"Your dataset needs one of the label columns for an evaluation: {self.output}"
            )

        instances = dataset.to_instances(self, lazy=lazy)
        instances.index_with(self.backbone.vocab)
        data_loader = create_dataloader(instances, batch_size=batch_size)

        try:
            return evaluate(
                self._model,
                data_loader,
                cuda_device=cuda_device,
                predictions_output_file=predictions_output_file,
                output_file=metrics_output_file,
            )
        finally:
            self._model.to(prior_device if prior_device >= 0 else "cpu")

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

    def model_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Returns an iterator over all model parameters, yielding the name and the parameter itself.

        Examples
        --------
        You can use this to freeze certain parameters in the training:

        >>> pipeline = Pipeline.from_config({
        ...     "name": "model_parameters_example",
        ...     "head": {"type": "TextClassification", "labels": ["a", "b"]},
        ... })
        >>> for name, parameter in pipeline.model_parameters():
        ...     if not name.endswith("bias"):
        ...         parameter.requires_grad = False

        """
        return self._model.named_parameters()

    def copy(self) -> "Pipeline":
        """Returns a copy of the pipeline"""
        model = PipelineModel.from_params(
            Params({"config": self.config}), vocab=copy.deepcopy(self.vocab)
        )
        config = copy.deepcopy(self._config)

        pipeline_copy = Pipeline(model, config)
        pipeline_copy._model.load_state_dict(self._model.state_dict())

        return pipeline_copy

    def save(self, directory: Union[str, Path]) -> str:
        """Saves the pipeline in the given directory as `model.tar.gz` file.

        Parameters
        ----------
        directory
            Save the 'model.tar.gz' file to this directory.

        Returns
        -------
        file_path
            Path to the 'model.tar.gz' file.
        """
        if isinstance(directory, str):
            directory = Path(directory)

        directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.vocab.save_to_files(str(temp_path / "vocabulary"))
            torch.save(self._model.state_dict(), temp_path / "best.th")
            with (temp_path / "config.json").open("w") as file:
                json.dump(
                    {
                        "model": {
                            "config": self.config.as_dict(),
                            "type": "PipelineModel",
                        }
                    },
                    file,
                    indent=4,
                )
            archive_model(temp_path, archive_path=directory)

        return str(directory / "model.tar.gz")

    def to_mlflow(
        self,
        tracking_uri: Optional[str] = None,
        experiment_id: Optional[int] = None,
        run_name: str = "Log biome.text model",
    ) -> str:
        """Logs the pipeline as MLFlow Model to a MLFlow Tracking server

        Parameters
        ----------
        tracking_uri
            The URI of the MLFlow tracking server. MLFlow defaults to './mlruns'.
        experiment_id
            ID of the experiment under which to create the logging run. If this argument is unspecified,
            will look for valid experiment in the following order: activated using `mlflow.set_experiment`,
            `MLFLOW_EXPERIMENT_NAME` environment variable, `MLFLOW_EXPERIMENT_ID` environment variable,
            or the default experiment as defined by the tracking server.
        run_name
            The name of the MLFlow run logging the model

        Returns
        -------
        model_uri
            The URI of the logged MLFlow model. The model gets logged as an artifact to the corresponding run.

        Examples
        --------
        After logging the pipeline to MLFlow you can use the MLFlow model for inference:
        >>> import mlflow, pandas, biome.text
        >>> pipeline = biome.text.Pipeline.from_config({
        ...     "name": "to_mlflow_example",
        ...     "head": {"type": "TextClassification", "labels": ["a", "b"]},
        ... })
        >>> model_uri = pipeline.to_mlflow()
        >>> model = mlflow.pyfunc.load_model(model_uri)
        >>> prediction: pandas.DataFrame = model.predict(pandas.DataFrame([{"text": "Test this text"}]))
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # This conda environment is only needed when serving the model later on with `mlflow models serve`
        conda_env = {
            "name": "mlflow-dev",
            "channels": ["defaults", "conda-forge"],
            "dependencies": ["python=3.7.9", "pip>=20.3.0", {"pip": ["biome-text"]}],
        }

        with tempfile.TemporaryDirectory() as tmpdir_name:
            file_path = self.save(directory=tmpdir_name)

            with mlflow.start_run(
                experiment_id=experiment_id, run_name=run_name
            ) as run:
                mlflow.pyfunc.log_model(
                    artifact_path="BiomeTextModel",
                    loader_module="biome.text.mlflow_model",
                    data_path=file_path,
                    conda_env=conda_env,
                )
                model_uri = os.path.join(run.info.artifact_uri, "BiomeTextModel")

        return model_uri

    @staticmethod
    def _add_transformers_vocab_if_needed(model: PipelineModel):
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
        transformers_indexer = model.head.backbone.featurizer.indexer.get(
            TransformersFeatures.namespace
        )
        if transformers_indexer is not None:
            try:
                transformers_indexer._add_encoding_to_vocabulary_if_needed(model.vocab)
            except AttributeError:
                transformers_indexer._matched_indexer._add_encoding_to_vocabulary_if_needed(
                    model.vocab
                )

    @staticmethod
    def _model_from_archive(archive: Archive) -> PipelineModel:
        if not isinstance(archive.model, PipelineModel):
            raise ValueError(f"Wrong pipeline model: {archive.model}")
        return cast(PipelineModel, archive.model)

    @staticmethod
    def _config_from_archive(archive: Archive) -> PipelineConfiguration:
        config = archive.config["model"]["config"]
        return PipelineConfiguration.from_params(config)

    # deprecated methods:

    def create_vocabulary(self, config: VocabularyConfiguration) -> None:
        """Creates the vocabulary for the pipeline from scratch

        DEPRECATED: The vocabulary is now created automatically and this method will be removed in the future.
        You can directly pass on a `VocabularyConfiguration` to the `train` method or use its default.

        Parameters
        ----------
        config
            Specifies the sources of the vocabulary and how to extract it
        """
        raise DeprecationWarning(
            "The vocabulary is created automatically and this method will be removed in the future. "
            "You can directly pass on a `VocabularyConfiguration` to the `train` method or use its default."
        )

    def predict_batch(self, *args, **kwargs):
        """DEPRECATED"""
        raise DeprecationWarning(
            "Use `self.predict(batch=...)` instead. This method will be removed in the future."
        )

    def explain(self, *args, **kwargs):
        """DEPRECATED"""
        raise DeprecationWarning(
            "Use `self.predict(..., add_attributions=True)` instead. This method will be removed in the future."
        )

    def explain_batch(self, *args, **kwargs):
        """DEPRECATED"""
        raise DeprecationWarning(
            "Use `self.predict(batch=..., add_attributions=True)` instead. This method will be removed in the future."
        )


class PredictionError(Exception):
    """Exception for a failed prediction of a single input or a whole batch"""

    # For now this is the only Error in this module. If you want to add further Errors
    # define a base class as recommended in https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
    pass
