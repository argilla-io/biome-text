import copy
import inspect
import json
import logging
import os
import tempfile
from inspect import Parameter
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
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
from allennlp.common.file_utils import is_url_or_existing_file
from allennlp.data import Vocabulary
from allennlp.models import load_archive
from allennlp.models.archival import Archive
from allennlp.models.archival import archive_model

from biome.text import __version__
from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.configuration import PipelineConfiguration
from biome.text.configuration import PredictionConfiguration
from biome.text.configuration import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.dataset import InstanceDataset
from biome.text.features import TransformersFeatures
from biome.text.features import WordFeatures
from biome.text.helpers import update_method_signature
from biome.text.mlflow_model import BiomeTextModel
from biome.text.model import PipelineModel
from biome.text.modules.heads import TaskHead
from biome.text.modules.heads import TaskHeadConfiguration
from biome.text.trainer import Trainer


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
    def from_yaml(cls, path: str) -> "Pipeline":
        """Creates a pipeline from a config yaml file

        Parameters
        ----------
        path
            The path to a YAML configuration file

        Returns
        -------
        pipeline
            A configured pipeline
        """
        pipeline_configuration = PipelineConfiguration.from_yaml(path)

        return cls.from_config(pipeline_configuration)

    @classmethod
    def from_config(
        cls,
        config: Union[PipelineConfiguration, dict],
    ) -> "Pipeline":
        """Creates a pipeline from a `PipelineConfiguration` object or a configuration dictionary

        Parameters
        ----------
        config
            A `PipelineConfiguration` object or a configuration dict

        Returns
        -------
        pipeline
            A configured pipeline
        """
        if isinstance(config, PipelineConfiguration):
            config = config.as_dict()

        model = PipelineModel(config=config)

        if not isinstance(model, PipelineModel):
            raise TypeError(f"Cannot load model. Wrong format of {model}")

        cls._add_transformers_vocab_if_needed(model)

        return cls(model, PipelineConfiguration.from_dict(config))

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

    def create_vocab(
        self,
        instance_datasets: Iterable[InstanceDataset],
        config: Optional[VocabularyConfiguration] = None,
    ) -> Vocabulary:
        """Creates and updates the vocab of the pipeline.

        NOTE: The trainer calls this method for you. You can use this method in case you want
        to create the vocab outside of the training process.

        Parameters
        ----------
        instance_datasets
            A list of instance datasets from which to create the vocabulary.
        config
            Configurations for the vocab creation. Default: `VocabularyConfiguration()`.

        Examples
        --------
        >>> from biome.text import Pipeline, Dataset
        >>> pl = Pipeline.from_config(
        ...     {"name": "example", "head":{"type": "TextClassification", "labels": ["pos", "neg"]}}
        ... )
        >>> dataset = Dataset.from_dict({"text": ["Just an example"], "label": ["pos"]})
        >>> instance_dataset = dataset.to_instances(pl)
        >>> vocab = pl.create_vocab([instance_dataset])
        """
        # The transformers feature comes with its own vocab, no need to create anything if it is the only feature
        if self.config.features.configured_namespaces == [
            TransformersFeatures.namespace
        ]:
            return self.vocab

        self._check_for_word_vector_weights_file()

        config = config or VocabularyConfiguration()

        vocab = Vocabulary.from_instances(
            instances=(
                instance for dataset in instance_datasets for instance in dataset
            ),
            max_vocab_size=config.max_vocab_size,
            min_count=config.min_count,
            pretrained_files=config.pretrained_files,
            only_include_pretrained_words=config.only_include_pretrained_words,
            min_pretrained_embeddings=config.min_pretrained_embeddings,
            tokens_to_add=config.tokens_to_add,
        )

        # If the vocab is the same, this is just a no-op
        self._model.extend_vocabulary(vocab)

        return vocab

    def _check_for_word_vector_weights_file(self):
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

    def _restore_vocab(self, folder: str) -> Vocabulary:
        # The transformers feature comes with its own vocab, no need to restore anything if it is the only feature
        if self.config.features.configured_namespaces == [
            TransformersFeatures.namespace
        ]:
            return self.vocab

        self._check_for_word_vector_weights_file()

        vocab = Vocabulary.from_files(folder)
        self._model.extend_vocabulary(vocab)

        return vocab

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
            If a prediction fails, its return value will be `None`.
        """
        if args or kwargs:
            batch = [self._map_args_kwargs_to_input(*args, **kwargs)]

        prediction_config = PredictionConfiguration(
            add_tokens=add_tokens,
            add_attributions=add_attributions,
            attributions_kwargs=attributions_kwargs or {},
        )

        predictions = self._model.predict(batch, prediction_config)

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
        test_dataset: Union[Dataset, InstanceDataset],
        batch_size: int = 16,
        lazy: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate your model on a test dataset

        Parameters
        ----------
        test_dataset
            The test data set.
        batch_size
            The batch size. Default: 16.
        lazy
            If True, instances are lazily loaded from disk, otherwise they are loaded into memory.
            Ignored when `test_dataset` is a `InstanceDataset`. Default: False.
        output_dir
            Save a `metrics.json` to this output directory. Default: None.
        verbose
            If True, prints the test results. Default: True.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the metrics
        """
        trainer = Trainer(self, lazy=lazy)

        return trainer.test(
            test_dataset, batch_size=batch_size, output_dir=output_dir, verbose=verbose
        )

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
        model = PipelineModel(self._config.as_dict(), vocab=copy.deepcopy(self.vocab))
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
        run_name: str = "log_biometext_model",
        input_example: Optional[Dict] = None,
        conda_env: Optional[Dict] = None,
    ) -> str:
        """Logs the pipeline as MLFlow Model to a MLFlow Tracking server

        Parameters
        ----------
        tracking_uri
            The URI of the MLFlow tracking server, MLFlow defaults to './mlruns'. Default: None
        experiment_id
            ID of the experiment under which to create the logging run. If this argument is unspecified,
            will look for valid experiment in the following order: activated using `mlflow.set_experiment`,
            `MLFLOW_EXPERIMENT_NAME` environment variable, `MLFLOW_EXPERIMENT_ID` environment variable,
            or the default experiment as defined by the tracking server.
        run_name
            The name of the MLFlow run logging the model. Default: 'log_biometext_model'.
        input_example
            You can provide an input example in the form of a dictionary. For example, for a TextClassification head
            this would be `{"text": "This is an input example"}`.
        conda_env
            This conda environment is used when serving the model via `mlflow models serve`. Default:
            conda_env = {
                "name": "mlflow-dev",
                "channels": ["defaults", "conda-forge"],
                "dependencies": ["python=3.7.9", "pip>=20.3.0", {"pip": ["biome-text=={__version__}"]}],
            }

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
        conda_env = conda_env or {
            "name": "mlflow-dev",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.7.9",
                "pip>=20.3.0",
                {"pip": [f"biome-text=={__version__}"]},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir_name:
            file_path = Path(self.save(directory=tmpdir_name))

            with mlflow.start_run(
                experiment_id=experiment_id, run_name=run_name
            ) as run:
                mlflow.log_artifact(str(file_path), "biometext_pipeline")
                mlflow.pyfunc.log_model(
                    artifact_path="mlflow_model",
                    python_model=BiomeTextModel(),
                    artifacts={
                        BiomeTextModel.ARTIFACT_CONTEXT: mlflow.get_artifact_uri(
                            f"biometext_pipeline/{file_path.name}"
                        )
                    },
                    input_example=input_example,
                    conda_env=conda_env,
                )
                model_uri = os.path.join(run.info.artifact_uri, "mlflow_model")

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
