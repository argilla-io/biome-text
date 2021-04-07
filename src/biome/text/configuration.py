import copy
import os
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import yaml
from allennlp.common import FromParams
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import TokenIndexer
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from pytorch_lightning import Callback
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.profiler import BaseProfiler

from biome.text.features import CharFeatures
from biome.text.features import TransformersFeatures
from biome.text.features import WordFeatures
from biome.text.featurizer import InputFeaturizer
from biome.text.helpers import sanitize_for_params
from biome.text.helpers import save_dict_as_yaml
from biome.text.modules.encoders import Encoder
from biome.text.modules.heads.classification.relation_classification import (
    RelationClassification,
)
from biome.text.modules.heads.task_head import TaskHeadConfiguration
from biome.text.modules.heads.token_classification import TokenClassification
from biome.text.tokenizer import Tokenizer
from biome.text.tokenizer import TransformersTokenizer


class FeaturesConfiguration(FromParams):
    """Configures the input features of the `Pipeline`

    Use this for defining the features to be used by the model, namely word and character embeddings.

    :::tip
    If you do not pass in either of the parameters (`word` or `char`),
    your pipeline will be setup with a default word feature (embedding_dim=50).
    :::

    Example:

    ```python
    word = WordFeatures(embedding_dim=100)
    char = CharFeatures(embedding_dim=16, encoder={'type': 'gru'})
    config = FeaturesConfiguration(word, char)
    ```

    Parameters
    ----------
    word
        The word feature configurations, see `biome.text.features.WordFeatures`
    char
        The character feature configurations, see `biome.text.features.CharFeatures`
    transformers
        The transformers feature configuration, see `biome.text.features.TransformersFeatures`
        A word-level representation of the [transformer](https://huggingface.co/models) models using AllenNLP's

    """

    __DEFAULT_CONFIG = WordFeatures(embedding_dim=50)

    def __init__(
        self,
        word: Optional[WordFeatures] = None,
        char: Optional[CharFeatures] = None,
        transformers: Optional[TransformersFeatures] = None,
    ):
        self.word = word
        self.char = char
        self.transformers = transformers

        if not (word or char or transformers):
            self.word = self.__DEFAULT_CONFIG

    @classmethod
    def from_params(
        cls: Type["FeaturesConfiguration"], params: Params, **extras
    ) -> "FeaturesConfiguration":

        word = params.pop("word", None)
        word = WordFeatures(**word.as_dict(quiet=True)) if word else None

        char = params.pop("char", None)
        char = CharFeatures(**char.as_dict(quiet=True)) if char else None

        transformers = params.pop("transformers", None)
        transformers = (
            TransformersFeatures(**transformers.as_dict(quiet=True))
            if transformers
            else None
        )

        params.assert_empty("FeaturesConfiguration")
        return cls(word=word, char=char, transformers=transformers)

    @property
    def configured_namespaces(self) -> List[str]:
        """Return the namespaces of the features that are configured"""
        return [feature.namespace for feature in self]

    def compile_embedder(self, vocab: Vocabulary) -> TextFieldEmbedder:
        """Creates the embedder based on the configured input features

        Parameters
        ----------
        vocab
            The vocabulary for which to create the embedder

        Returns
        -------
        embedder
        """
        configuration = self._make_allennlp_config()

        # We have to set the weights_file / pretrained_file (pretrained word vectors) to None for two reasons:
        # - compiling the embedder with a weights file and an empty vocab will fail,
        #   this is the case for untrained pipelines
        # - compiling the embedder with a non existent weights file will fail,
        #   but for pretrained pipelines it should be optional
        # We will "reactivate" it after the embedder is initialized for extending the vocab with it.
        try:
            configuration["word"]["embedder"]["pretrained_file"] = None
        except KeyError:
            pass

        text_field_embedder = TextFieldEmbedder.from_params(
            Params(
                {
                    "token_embedders": {
                        feature_namespace: config["embedder"]
                        for feature_namespace, config in configuration.items()
                    }
                }
            ),
            vocab=vocab,
        )

        # It is save to reactivate it, since extending the vocab will only throw a warning if the file does not exist.
        if self.word is not None:
            setattr(
                getattr(text_field_embedder, f"token_embedder_{self.word.namespace}"),
                "_pretrained_file",
                self.word.weights_file,
            )

        return text_field_embedder

    def compile_featurizer(self, tokenizer: Tokenizer) -> InputFeaturizer:
        """Creates the featurizer based on the configured input features

        :::tip
        If you are creating configurations programmatically
        use this method to check that you provided a valid configuration.
        :::

        Parameters
        ----------
        tokenizer
            Tokenizer used for this featurizer

        Returns
        -------
        featurizer
            The configured `InputFeaturizer`
        """
        configuration = self._make_allennlp_config()

        indexer = {
            feature_namespace: TokenIndexer.from_params(Params(config["indexer"]))
            for feature_namespace, config in configuration.items()
        }

        return InputFeaturizer(tokenizer, indexer=indexer)

    def _make_allennlp_config(self) -> Dict[str, Any]:
        """Returns a configuration dict compatible with allennlp

        Returns
        -------
        config_dict
        """
        configuration = {feature.namespace: feature.config for feature in self}

        return copy.deepcopy(configuration)

    def __iter__(self):
        for feature in [self.word, self.char, self.transformers]:
            if feature is not None:
                yield feature


class TokenizerConfiguration(FromParams):
    """Configures the `Tokenizer`

    Parameters
    ----------
    lang
        The [spaCy model used](https://spacy.io/api/tokenizer) for tokenization is language dependent.
        For optimal performance, specify the language of your input data (default: "en").
    max_sequence_length
        Maximum length in characters for input texts truncated with `[:max_sequence_length]` after `TextCleaning`.
    max_nr_of_sentences
        Maximum number of sentences to keep when using `segment_sentences` truncated with `[:max_sequence_length]`.
    text_cleaning
        A `TextCleaning` configuration with pre-processing rules for cleaning up and transforming raw input text.
    segment_sentences
        Whether to segment input texts into sentences.
    use_spacy_tokens
        If True, the tokenized token list contains spacy tokens instead of allennlp tokens
    remove_space_tokens
        If True, all found space tokens will be removed from the final token list.
    start_tokens
        A list of token strings to the sequence before tokenized input text.
    end_tokens
        A list of token strings to the sequence after tokenized input text.
    use_transformers
        If true, we will use a transformers tokenizer from HuggingFace and disregard all other parameters above.
        If you specify any of the above parameters you want to set this to false.
        If None, we automatically choose the right value based on your feature and head configuration.
    transformers_kwargs
        This dict is passed on to AllenNLP's `PretrainedTransformerTokenizer`.
        If no `model_name` key is provided, we will infer one from the features configuration.
    """

    # note: It's important that it inherits from FromParas so that `Pipeline.from_pretrained()` works!
    def __init__(
        self,
        lang: str = "en",
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
        text_cleaning: Optional[Dict[str, Any]] = None,
        segment_sentences: bool = False,
        use_spacy_tokens: bool = False,
        remove_space_tokens: bool = True,
        start_tokens: Optional[List[str]] = None,
        end_tokens: Optional[List[str]] = None,
        use_transformers: Optional[bool] = None,
        transformers_kwargs: Optional[Dict] = None,
    ):
        self.lang = lang
        self.max_sequence_length = max_sequence_length
        self.max_nr_of_sentences = max_nr_of_sentences
        self.segment_sentences = segment_sentences
        self.text_cleaning = text_cleaning
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.use_spacy_tokens = use_spacy_tokens
        self.remove_space_tokens = remove_space_tokens
        self.use_transformers = use_transformers
        self.transformers_kwargs = transformers_kwargs or {}

    def __eq__(self, other):
        return all(a == b for a, b in zip(vars(self), vars(other)))


class PipelineConfiguration(FromParams):
    """Creates a `Pipeline` configuration

    Parameters
    ----------
    name
        The `name` for our pipeline
    features
        The input `features` to be used by the model pipeline. We define this using a `FeaturesConfiguration` object.
    head
        The `head` for the task, e.g., a LanguageModelling task, using a `TaskHeadConfiguration` object.
    tokenizer
        The `tokenizer` defined with a `TokenizerConfiguration` object.
    encoder
        The core text seq2seq `encoder` of our model using a `Seq2SeqEncoderConfiguration`
    """

    # To be able to skip the configuration checks when testing
    _SKIP_CHECKS = False

    def __init__(
        self,
        name: str,
        head: TaskHeadConfiguration,
        features: Optional[FeaturesConfiguration] = None,
        tokenizer: Optional[TokenizerConfiguration] = None,
        encoder: Optional[Encoder] = None,
    ):
        super(PipelineConfiguration, self).__init__()

        self.name = name
        self.head = head
        self.features = features or FeaturesConfiguration()
        self.tokenizer_config = tokenizer or TokenizerConfiguration()

        # figure out if we need a transformers tokenizer
        if self.tokenizer_config.use_transformers is None:
            self._use_transformers_tokenizer_if_sensible()

        # make sure we use the right indexer/embedder for the transformers feature
        if self.tokenizer_config.use_transformers:
            self.features.transformers.mismatched = False
            if self.tokenizer_config.transformers_kwargs.get("model_name") is None:
                self.tokenizer_config.transformers_kwargs[
                    "model_name"
                ] = self.features.transformers.model_name

        if not self._SKIP_CHECKS:
            self._check_for_incompatible_configurations()

        self.encoder = encoder

    def _use_transformers_tokenizer_if_sensible(self):
        if (
            # Only use word pieces if no word-based feature was chosen
            self.features.transformers is not None
            and self.features.word is None
            and self.features.char is None
            # NER tags are usually given per word not per word pieces
            and TokenClassification.__name__ not in self.head.config["type"]
            and RelationClassification.__name__ not in self.head.config["type"]
        ):
            self.tokenizer_config.use_transformers = True
            self.tokenizer_config.transformers_kwargs = (
                self.tokenizer_config.transformers_kwargs
                or {"model_name": self.features.transformers.model_name}
            )
        else:
            self.tokenizer_config.use_transformers = False

    def _check_for_incompatible_configurations(self):
        if self.tokenizer_config.use_transformers:
            if self.features.word is not None or self.features.char is not None:
                raise ConfigurationError(
                    "You are trying to use word or char features on subwords and possibly special tokens."
                    "This is not recommended!"
                )

            if (
                TokenClassification.__name__ in self.head.config["type"]
                or RelationClassification.__name__ in self.head.config["type"]
            ):
                raise NotImplementedError(
                    "You specified a transformers tokenizer, but the 'TokenClassification' and "
                    "'RelationClassification' heads are still not capable of dealing with subword/special tokens."
                )

            if (
                self.tokenizer_config.transformers_kwargs["model_name"]
                != self.features.transformers.model_name
            ):
                raise ConfigurationError(
                    f"The model_name of the TransformerTokenizer "
                    f"({self.tokenizer_config.transformers_kwargs['model_name']}) does not match the model_name of "
                    f"your transformers feature ({self.features.transformers.model_name})!"
                )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfiguration":
        """Creates a pipeline configuration from a config yaml file

        Parameters
        ----------
        path
            The path to a YAML configuration file

        Returns
        -------
        pipeline_configuration
        """
        with open(path) as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfiguration":
        """Creates a pipeline configuration from a config dictionary

        Parameters
        ----------
        config_dict
            A configuration dictionary

        Returns
        -------
        pipeline_configuration
        """
        config_dict = sanitize_for_params(copy.deepcopy(config_dict))
        return PipelineConfiguration.from_params(Params(config_dict))

    def as_dict(self) -> Dict[str, Any]:
        """Returns the configuration as dictionary

        Returns
        -------
        config
        """
        config = {
            "name": self.name,
            "tokenizer": vars(self.tokenizer_config),
            "features": vars(self.features),
            "head": self.head.config,
        }

        if self.encoder:
            config["encoder"] = self.encoder.config

        return sanitize_for_params(config)

    def to_yaml(self, path: str):
        """Saves the pipeline configuration to a yaml formatted file

        Parameters
        ----------
        path
            Path to the output file
        """
        config_dict = copy.deepcopy(self.as_dict())
        for feature_name, feature in config_dict["features"]:
            if feature is not None:
                config_dict["features"][feature_name] = feature.to_json()

        save_dict_as_yaml(config_dict, path)

    def build_tokenizer(self) -> Tokenizer:
        """Build the pipeline tokenizer"""
        if self.tokenizer_config.use_transformers:
            return TransformersTokenizer(self.tokenizer_config)
        return Tokenizer(self.tokenizer_config)

    def build_featurizer(self) -> InputFeaturizer:
        """Creates the pipeline featurizer"""
        return self.features.compile_featurizer(self.build_tokenizer())

    def build_embedder(self, vocab: Vocabulary):
        """Build the pipeline embedder for aiven dictionary"""
        return self.features.compile_embedder(vocab)


@dataclass
class TrainerConfiguration:
    """Configures the training of a pipeline

    It is passed on to the `Pipeline.train` method. Doc strings mainly provided by
    [AllenNLP](https://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects)

    Attributes
    ----------
    optimizer
        [Pytorch optimizers](https://pytorch.org/docs/stable/optim.html)
        that can be constructed via the AllenNLP configuration framework
    validation_metric
        Validation metric to measure for whether to stop training using patience
        and whether to serialize an is_best model each epoch.
        The metric name must be prepended with either "+" or "-",
        which specifies whether the metric is an increasing or decreasing function.
    patience
        Number of epochs to be patient before early stopping:
        the training is stopped after `patience` epochs with no improvement.
        If given, it must be > 0. If `None`, early stopping is disabled.
    num_epochs
        Number of training epochs
    cuda_device
        An integer specifying the CUDA device to use for this process. If -1, the CPU is used.
        By default (None) we will automatically use a CUDA device if one is available.
    grad_norm
        If provided, gradient norms will be rescaled to have a maximum of this value.
    grad_clipping
        If provided, gradients will be clipped during the backward pass to have an (absolute) maximum of this value.
        If you are getting `NaN`s in your gradients during training that are not solved by using grad_norm,
        you may need this.
    learning_rate_scheduler
        If specified, the learning rate will be decayed with respect to this schedule at the end of each epoch
        (or batch, if the scheduler implements the step_batch method).
        If you use `torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the `validation_metric` provided
        to determine if learning has plateaued.
    warmup_steps
        The number of warmup steps. This parameter is ignored if a `learning_rate_scheduler` is provided.
    linear_decay
        If true, linearly decrease the learning rate to 0 until the end of the training. See also `training_size` below!
        This parameter is ignored if a `learning_rate_scheduler` is provided.
    training_size
        If you set `linear_decay` to true and work with lazily loaded training data,
        you need to provide the number of examples in your training data.
    momentum_scheduler
        If specified, the momentum will be updated at the end of each batch or epoch according to the schedule.
    moving_average
        If provided, we will maintain moving averages for all parameters.
        During training, we employ a shadow variable for each parameter, which maintains the moving average.
        During evaluation, we backup the original parameters and assign the moving averages to corresponding parameters.
        Be careful that when saving the checkpoint, we will save the moving averages of parameters.
        This is necessary because we want the saved model to perform as well as the validated model if we load it later.
    batch_size
        Size of the batch.
    data_bucketing
        If enabled, try to apply data bucketing over training batches.
    batches_per_epoch
        Determines the number of batches after which a training epoch ends.
        If the number is smaller than the total amount of batches in your training data,
        the second "epoch" will take off where the first "epoch" ended.
        If this is `None`, then an epoch is set to be one full pass through your training data.
        This is useful if you want to evaluate your data more frequently on your validation data set during training.
    random_seed
        Seed for the underlying random number generators.
        If None, we take the random seeds provided by AllenNLP's `prepare_environment` method.
    use_amp
        If `True`, we'll train using [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html).
    num_serialized_models_to_keep
        Number of previous model checkpoints to retain.  Default is to keep 1 checkpoint.
        A value of None or -1 means all checkpoints will be kept.
    """

    optimizer: Dict[str, Any] = field(default_factory=lambda: {"type": "adam"})
    validation_metric: str = "-loss"
    patience: Optional[int] = 2
    num_epochs: int = 20
    cuda_device: Optional[int] = None
    grad_norm: Optional[float] = None
    grad_clipping: Optional[float] = None
    learning_rate_scheduler: Optional[Dict[str, Any]] = None
    warmup_steps: int = 0
    linear_decay: bool = False
    training_size: Optional[int] = None
    momentum_scheduler: Optional[Dict[str, Any]] = None
    moving_average: Optional[Dict[str, Any]] = None
    use_amp: bool = False
    num_serialized_models_to_keep: int = 1
    # Data loader parameters
    batch_size: Optional[int] = 16
    data_bucketing: bool = False
    batches_per_epoch: Optional[int] = None
    # prepare_environment
    random_seed: Optional[int] = None

    def to_allennlp_trainer(self) -> Dict[str, Any]:
        """Returns a configuration dict formatted for AllenNLP's trainer

        Returns
        -------
        allennlp_trainer_config
        """
        # config for AllenNLP's GradientDescentTrainer
        allennlp_trainer_config = {
            "optimizer": self.optimizer,
            "patience": self.patience,
            "validation_metric": self.validation_metric,
            "num_epochs": self.num_epochs,
            "checkpointer": {
                "num_serialized_models_to_keep": self.num_serialized_models_to_keep
            },
            "cuda_device": self.cuda_device,
            "grad_norm": self.grad_norm,
            "grad_clipping": self.grad_clipping,
            "learning_rate_scheduler": self.learning_rate_scheduler,
            "tensorboard_writer": {"should_log_learning_rate": True},
            "moving_average": self.moving_average,
            "use_amp": self.use_amp,
        }

        return allennlp_trainer_config


@dataclass
class VocabularyConfiguration:
    """Configurations for creating the vocabulary

    See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])

    Parameters
    ----------
    include_valid_data
        If passed to the `Trainer`, this argument allows you to take the validation data into account when creating
        the vocabulary (apart from the training data). Default: False.
    max_vocab_size
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        `counter` can have a separate maximum vocabulary size. Any missing key will have a value
        of `None`, which means no cap on the vocabulary size.
    min_count
        Minimum number of appearances of a token to be included in the vocabulary.
        The key in the dictionary refers to the namespace of the input feature
    min_pretrained_embeddings
        Minimum number of lines to keep from pretrained_files, even for tokens not appearing in the sources.
    only_include_pretrained_words
        Only include tokens present in pretrained_files
    pretrained_files
        If provided, this map specifies the path to optional pretrained embedding files for each
        namespace. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of `only_include_pretrained_words`.
        Words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.
    tokens_to_add
        A list of tokens to add to the corresponding namespace of the vocabulary,
        even if they are not present in the `datasets`
    """

    include_valid_data: bool = False
    max_vocab_size: Union[int, Dict[str, int]] = None
    min_count: Dict[str, int] = None
    min_pretrained_embeddings: Dict[str, int] = None
    only_include_pretrained_words: bool = False
    pretrained_files: Optional[Dict[str, str]] = None
    tokens_to_add: Dict[str, List[str]] = None


@dataclass
class FindLRConfiguration:
    """A configuration for finding the learning rate via `Pipeline.find_lr()`.

    The `Pipeline.find_lr()` method increases the learning rate from `start_lr` to `end_lr` recording the losses.

    Parameters
    ----------
    start_lr
        The learning rate to start the search.
    end_lr
        The learning rate upto which search is done.
    num_batches
        Number of batches to run the learning rate finder.
    linear_steps
        Increase learning rate linearly if False exponentially.
    stopping_factor
        Stop the search when the current loss exceeds the best loss recorded by
        multiple of stopping factor. If `None` search proceeds till the `end_lr`
    """

    start_lr: float = 1e-5
    end_lr: float = 10
    num_batches: int = 100
    linear_steps: bool = False
    stopping_factor: Optional[float] = None


# We need this to be hashable for the prediction cache -> frozen=True
@dataclass(frozen=True)
class PredictionConfiguration:
    """Contains configurations for a `Pipeline.prediction`

    Parameters
    ----------
    add_tokens
    add_attributions
    attributions_kwargs
    """

    add_tokens: bool = False
    add_attributions: bool = False
    attributions_kwargs: Dict = field(default_factory=dict)


@dataclass
class LightningTrainerConfiguration:
    """Configuration for our Lightning Trainer

    The docs are mainly a copy from the
    [Lightning Trainer API](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer)
    with some additional parameters added.

    Parameters
    ----------
    accumulate_grad_batches
        Accumulates grads every k batches or as set up in the dict.

    add_early_stopping
        Adds a default `EarlyStopping` callback. To configure it, see `patience`, `monitor` and `monitor_mode`.
        Default: True

    add_csv_logger
        Adds a default CSV logger if `logger` is not False. Default: True

    add_tensorboard_logger
        Adds a default Tensorboard logger if `logger` is not False. Default: True

    add_wandb_logger
        Adds a default WandB logger if `logger` is not False and wandb is installed. Default: True

    auto_lr_find
        If set to True, will make trainer.tune() run a learning rate finder,
        trying to optimize initial learning for faster convergence. trainer.tune() method will
        set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
        To use a different key set a string instead of True with the key name.

    auto_scale_batch_size
        If set to True, will `initially` run a batch size
        finder trying to find the largest batch size that fits into memory.
        The result will be stored in self.batch_size in the LightningModule.
        Additionally, can be set to either `power` that estimates the batch size through
        a power search or `binsearch` that estimates the batch size through a binary search.

    batch_size
        Size of the batch.

    callbacks
        Add a callback or list of callbacks.

    checkpoint_callback
        Adds a default `ModelCheckpointWithVocab` callback if there is no user-defined ModelCheckpoint in
        `callbacks`. To configure it, see `save_top_k_checkpoints`, `monitor` and `monitor_mode`.
        Default: True.

    check_val_every_n_epoch
        Check val every n train epochs.

    data_bucketing
        If enabled, try to apply data bucketing over training batches.

    default_root_dir
        Default path for logs and weights when no logger/ckpt_callback passed.
        Can be remote file paths such as 's3://mybucket/path' or 'hdfs://path/'
        Default: './training_logs'.

    fast_dev_run
        runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
        of train, val and test to find any bugs (ie: a sort of unit test).

    flush_logs_every_n_steps
        How often to flush logs to disk (defaults to every 100 steps).

    gpus
        number of gpus to train on (int) or which GPUs to train on (list or str) applied per node.
        If None (default), we will use one GPU if available.

    gradient_clip_val
        0 means don't clip.

    limit_train_batches
        How much of training dataset to check (floats = percent, int = num_batches)

    limit_val_batches
        How much of validation dataset to check (floats = percent, int = num_batches)

    limit_test_batches
        How much of test dataset to check (floats = percent, int = num_batches)

    log_every_n_steps
        How often to log within steps (defaults to every 50 steps).

    logger
        Logger (or iterable collection of loggers) for experiment tracking.
        If not False, we will add some loggers by default, see `add_[csv, tensorboard, wandb]_logger`.
        Default: True

    max_epochs
        Stop training once this number of epochs is reached. Disabled by default (None).
        If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.

    min_epochs
        Force training for at least these many epochs. Disabled by default (None).
        If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.

    max_steps
        Stop training after this number of steps. Disabled by default (None).

    min_steps
        Force training for at least these number of steps. Disabled by default (None)

    monitor
        Metric to monitor. Will be used to load the best weights after the training.
        Has no effect if `checkpoint_callback` is False. Default: 'validation_loss'.

    monitor_mode
        Either 'min' or 'max'. If `save_top_k_checkpoints != 0`, the decision to overwrite the current save file is made
        based on either the maximization or the minimization of the monitored metric. Default: 'min'.

    num_sanity_val_steps
        Sanity check runs n validation batches before starting the training routine.
        Set it to `-1` to run all batches in all validation dataloaders. Default: 2

    optimizer
        Configuration for an [AllenNLP/PyTorch optimizer](https://docs.allennlp.org/main/api/training/optimizers/)
        that is constructed via the AllenNLP configuration framework.
        Default: `{"type": "adam", "lr": 0.001}`

    overfit_batches
        Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0

    patience
        Number of validation checks with no improvement after which training will be stopped. Default: 3.

    precision
        Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.

    progress_bar_refresh_rate
        How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
        Ignored when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
        a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).

    resume_from_checkpoint
        Path/URL of the checkpoint from which training is resumed. If there is
        no checkpoint file at the path, start from scratch. If resuming from mid-epoch checkpoint,
        training will start from the beginning of the next epoch.

    save_top_k_checkpoints
        If `save_top_k_checkpoints == k`, the best k models according to the metric monitored will be saved.
        If `save_top_k_checkpoints == 0`, no models are saved. If `save_top_k_checkpoints == -1`, all models are saved.
        Has no effect if `checkpoint_callback` is False. Default: 1.

    stochastic_weight_avg
        Whether to use `Stochastic Weight Averaging (SWA)
        <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_`

    terminate_on_nan
        If set to True, will terminate training (by raising a `ValueError`) at the
        end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

    val_check_interval
        How often to check the validation set. Use float to check within a training epoch,
        use int to check every n steps (batches).

    weights_save_path
        Where to save weights if specified. Will override default_root_dir
        for checkpoints only. Use this if for whatever reason you need the checkpoints
        stored in a different place than the logs written in `default_root_dir`.
        Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
        Defaults to `default_root_dir`.

    extra_lightning_params
        This dictionary is passed on as kwargs to the Pytorch Lightning Trainer init method.
    """

    accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1
    auto_lr_find: Union[bool, str] = False
    auto_scale_batch_size: Union[str, bool] = False
    callbacks: Optional[Union[List[Callback], Callback]] = None
    check_val_every_n_epoch: int = 1
    checkpoint_callback: bool = True
    default_root_dir: str = field(
        default_factory=lambda: os.path.join(os.getcwd(), "training_logs")
    )
    fast_dev_run: Union[int, bool] = False
    flush_logs_every_n_steps: int = 100
    gpus: Optional[Union[List[int], str, int]] = None
    gradient_clip_val: float = 0
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    log_every_n_steps: int = 50
    logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    num_sanity_val_steps: int = 2
    overfit_batches: Union[int, float] = 0.0
    precision: int = 32
    progress_bar_refresh_rate: Optional[int] = None
    resume_from_checkpoint: Optional[Union[Path, str]] = None
    stochastic_weight_avg: bool = False
    terminate_on_nan: bool = False
    val_check_interval: Union[int, float] = 1.0
    weights_save_path: Optional[str] = None
    # non lightning trainer parameters
    add_early_stopping: bool = True
    add_csv_logger: bool = True
    add_tensorboard_logger: bool = True
    add_wandb_logger: bool = True
    batch_size: int = 16
    data_bucketing: bool = False
    monitor: str = "validation_loss"
    monitor_mode: str = "min"
    optimizer: Dict[str, Any] = field(
        default_factory=lambda: {"type": "adam", "lr": 0.001}
    )
    patience: int = 3
    save_top_k_checkpoints: int = 1
    extra_lightning_params: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        """Returns the dataclass as dict without a deepcopy, in contrast to `dataclasses.asdict`"""
        return {fld.name: getattr(self, fld.name) for fld in fields(self)}

    @property
    def lightning_params(self) -> Dict[str, Any]:
        non_lightning_params = [
            "add_early_stopping",
            "add_csv_logger",
            "add_tensorboard_logger",
            "add_wandb_logger",
            "batch_size",
            "data_bucketing",
            "monitor",
            "monitor_mode",
            "optimizer",
            "patience",
            "save_top_k_checkpoints",
            "extra_lightning_params",
        ]

        lightning_params = {
            key: value
            for key, value in self.as_dict().items()
            if key not in non_lightning_params
        }
        lightning_params.update(self.extra_lightning_params)

        return lightning_params
