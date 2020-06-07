import copy
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from allennlp.common import FromParams, Params
from allennlp.data import TokenIndexer, Vocabulary
from allennlp.modules import TextFieldEmbedder

from biome.text.data import DataSource
from . import vocabulary
from .features import CharFeatures, WordFeatures
from .featurizer import InputFeaturizer
from .helpers import save_dict_as_yaml
from .modules.encoders import Encoder
from .modules.heads.task_head import TaskHeadSpec
from .tokenizer import Tokenizer


class FeaturesConfiguration(FromParams):
    """Creates a input featurizer configuration
    
    This class will create a configuration for the features of the `Pipeline`.
    
    Use this for defining the main features to be used by the model, namely word and character embeddings.
    
    :::tip
    If you do not pass `words` and `chars` your pipeline will be setup with default word features (embedding_dim=50).
    :::
    
    Example:
    
    ```python
    word = WordFeatures(embedding_dim=100)
    char = CharFeatures(embedding_dim=16, encoder={'type': 'gru'})
    config = FeaturesConfiguration(word, char)
    ```
    
    Parameters
    ----------
    word : `biome.text.features.WordFeatures`
    char: `biome.text.features.CharFeatures`
    extra_params
    """

    __DEFAULT_CONFIG = WordFeatures(embedding_dim=50)

    def __init__(
        self,
        word: Optional[WordFeatures] = None,
        char: Optional[CharFeatures] = None,
        **extra_params
    ):
        self.word = word or None
        self.char = char or None

        for k, v in extra_params.items():
            self.__setattr__(k, v)

        if not (word or char or extra_params):
            self.word = self.__DEFAULT_CONFIG

    @classmethod
    def from_params(
        cls: Type["FeaturesConfiguration"], params: Params, **extras
    ) -> "FeaturesConfiguration":

        word = params.pop("word", params.pop("words", None))  # TODO: remove backward
        word = WordFeatures(**word.as_dict(quiet=True)) if word else None

        char = params.pop("char", params.pop("chars", None))  # TODO: remove backward
        char = CharFeatures(**char.as_dict(quiet=True)) if char else None

        return cls(word, char, **params.as_dict(), **extras)

    @property
    def keys(self) -> List[str]:
        """Gets the key features"""
        return [key for key in vars(self)]

    def compile_embedder(self, vocab: Vocabulary) -> TextFieldEmbedder:
        """Creates the embedder from configured features for a given vocabulary"""
        configuration = self._make_allennlp_config()
        if vocabulary.is_empty(vocab, namespaces=self.keys) and configuration.get(
            "word"
        ):
            # We simplify embedder configuration for better load an blank pipeline which create the vocab
            embedder_cfg = configuration["word"]["embedder"]
            embedder_cfg["embedding_dim"] = 1
            if "pretrained_file" in embedder_cfg:
                embedder_cfg["pretrained_file"] = None
        return TextFieldEmbedder.from_params(
            Params(
                {
                    "token_embedders": {
                        feature: config["embedder"]
                        for feature, config in configuration.items()
                    }
                }
            ),
            vocab=vocab,
        )

    def compile_featurizer(self, tokenizer: Tokenizer) -> InputFeaturizer:
        """Creates a featurizer from the configuration object
        
        :::tip
        
        If you are creating configurations programmatically use this method to check that your config object contains
        a valid configuration.
        
        :::

        Parameters
        ----------
        tokenizer: `Tokenizer`
            tokenizer used for this featurizer

        Returns
        -------
        The configured `InputFeaturizer`
        """
        configuration = self._make_allennlp_config()

        indexer = {
            feature: TokenIndexer.from_params(Params(config["indexer"]))
            for feature, config in configuration.items()
        }
        return InputFeaturizer(tokenizer, indexer=indexer)

    def _make_allennlp_config(self) -> Dict[str, Any]:
        """Creates compatible allennlp configuration"""
        configuration = {k: v for k, v in vars(self).items() if isinstance(v, dict)}
        configuration.update(
            {spec.namespace: spec.config for spec in [self.word, self.char] if spec}
        )
        return copy.deepcopy(configuration)


class TokenizerConfiguration(FromParams):
    """Creates a `Tokenizer` configuration

    Parameters
    ----------
    lang
    skip_empty_tokens
    max_sequence_length
    max_nr_of_sentences
    text_cleaning
    segment_sentences
    """

    def __init__(
        self,
        lang: str = "en",
        skip_empty_tokens: bool = False,
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
        text_cleaning: Optional[Dict[str, Any]] = None,
        segment_sentences: Union[bool, Dict[str, Any]] = False,
    ):
        self.lang = lang
        self.skip_empty_tokens = skip_empty_tokens
        self.max_sequence_length = max_sequence_length
        self.max_nr_of_sentences = max_nr_of_sentences
        self.text_cleaning = text_cleaning
        self.segment_sentences = segment_sentences

    def compile(self) -> Tokenizer:
        """Build tokenizer object from its configuration"""
        return Tokenizer.from_params(Params(copy.deepcopy(vars(self))))


class PipelineConfiguration(FromParams):
    """"Creates a `Pipeline` configuration

    Parameters
    ----------
    name : `str`
        The `name` for our pipeline
    features : `FeaturesConfiguration`
        The input `features` to be used by the model pipeline. We define this using a `FeaturesConfiguration` object.
    head : `TaskHeadSpec`
        The `head` for the task, e.g., a LanguageModelling task, using a `TaskHeadSpec` object.
    tokenizer : `TokenizerConfiguration`, optional
        The `tokenizer` defined with a `TokenizerConfiguration` object.
    encoder : `Seq2SeqEncoderSpec`
        The core text seq2seq `encoder` of our model using a `Seq2SeqEncoderSpec`
    """

    def __init__(
        self,
        name: str,
        features: FeaturesConfiguration,
        head: TaskHeadSpec,
        tokenizer: Optional[TokenizerConfiguration] = None,
        encoder: Optional[Encoder] = None,
    ):
        super(PipelineConfiguration, self).__init__()

        self.name = name
        self.tokenizer = tokenizer or TokenizerConfiguration()
        self.features = features
        self.encoder = encoder
        self.head = head

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfiguration":
        """Creates a pipeline configuration from a config yaml file

        Parameters
        ----------
        path: `str`
            The path to a YAML configuration file

        Returns
        -------
        pipeline_configuration: `PipelineConfiguration`
        """
        with open(path) as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfiguration":
        """Creates a pipeline configuration from a config dictionary

        Parameters
        ----------
        config_dict: `dict`
            A configuration dictionary

        Returns
        -------
        pipeline_configuration: `PipelineConfiguration`
        """
        return PipelineConfiguration.from_params(Params(config_dict))

    def as_dict(self) -> Dict[str, Any]:
        config = {
            "name": self.name,
            "tokenizer": vars(self.tokenizer),
            "features": vars(self.features),
            "head": self.head.config,
        }

        if self.encoder:
            config["encoder"] = self.encoder.config

        return config

    def to_yaml(self, path: str):
        """Saves the pipeline configuration to a yaml formatted file

        Parameters
        ----------
        path : str
            Path to the output file
        """
        config_dict = copy.deepcopy(self.as_dict())
        config_dict["features"]["word"] = (
            config_dict["features"]["word"].to_dict()
            if config_dict["features"]["word"] is not None
            else None
        )
        config_dict["features"]["char"] = (
            config_dict["features"]["char"].to_dict()
            if config_dict["features"]["char"] is not None
            else None
        )

        save_dict_as_yaml(config_dict, path)

    def build_tokenizer(self) -> Tokenizer:
        """Build the pipeline tokenizer"""
        return self.tokenizer.compile()

    def build_featurizer(self) -> InputFeaturizer:
        """Creates the pipeline featurizer"""
        return self.features.compile_featurizer(self.tokenizer.compile())

    def build_embedder(self, vocab: Vocabulary):
        """Build the pipeline embedder for aiven dictionary"""
        return self.features.compile_embedder(vocab)


class TrainerConfiguration:
    """ Creates a `TrainerConfiguration`

    Parameters
    ----------
    optimizer
    validation_metric
    patience
    num_epochs
    cuda_device
    grad_norm
    grad_clipping
    learning_rate_scheduler
    momentum_scheduler
    moving_average
    batch_size
    cache_instances
    in_memory_batches
    data_bucketing
    """

    def __init__(
        self,
        optimizer: Dict[str, Any] = None,
        validation_metric: str = "-loss",
        patience: Optional[int] = 2,
        num_epochs: int = 20,
        cuda_device: int = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[Dict[str, Any]] = None,
        momentum_scheduler: Optional[Dict[str, Any]] = None,
        moving_average: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = 16,
        cache_instances: bool = True,
        in_memory_batches: int = 2,
        data_bucketing: bool = True,
    ):
        self.optimizer = optimizer or {"type": "adam"}
        self.validation_metric = validation_metric
        self.patience = patience
        self.num_epochs = num_epochs
        self.cuda_device = cuda_device
        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping
        self.learning_rate_scheduler = learning_rate_scheduler
        self.momentum_scheduler = momentum_scheduler
        self.moving_average = moving_average

        # Data Iteration
        self.batch_size = batch_size
        self.data_bucketing = data_bucketing
        self.cache_instances = cache_instances
        self.in_memory_batches = in_memory_batches


class VocabularyConfiguration:
    """Configures a ``Vocabulary`` before it gets created from data

    Use this to configure a Vocabulary using specific arguments from `allennlp.data.Vocabulary``

    See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])

    Parameters
    ----------
    sources : `List[DataSource]`
        Datasource to be used for data creation
    min_count : `Dict[str, int]`, optional (default=None)
        Minimum number of appearances of a token to be included in the vocabulary
    max_vocab_size :  `Union[int, Dict[str, int]]`, optional (default=`None`)
        Maximum number of tokens of the vocabulary
    pretrained_files : `Optional[Dict[str, str]]`, optional
        Pretrained files with word vectors
    only_include_pretrained_words : `bool`, optional (default=False)
        Only include tokens present in pretrained_files
    tokens_to_add : `Dict[str, int]`, optional
        A list of tokens to add to the vocabulary, even if they are not present in the ``sources``
    min_pretrained_embeddings : ``Dict[str, int]``, optional
        Minimum number of lines to keep from pretrained_files, even for tokens not appearing in the sources.
    """

    def __init__(
        self,
        sources: List[DataSource],
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
    ):
        self.sources = sources
        self.pretrained_files = pretrained_files
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.only_include_pretrained_words = only_include_pretrained_words
        self.tokens_to_add = tokens_to_add
        self.min_pretrained_embeddings = min_pretrained_embeddings
