import copy
from typing import Any, Dict, Optional, Type, Union, List

from allennlp.common import FromParams, Params

from biome.text.api_new.featurizer import InputFeaturizer
from biome.text.api_new.modules.encoders import Encoder
from biome.text.api_new.modules.heads.defs import TaskHeadSpec
from biome.text.api_new.tokenizer import Tokenizer


class FeaturesConfiguration(FromParams):
    """Features configuration spec"""

    def __init__(
        self,
        words: Optional[Dict[str, Any]] = None,
        chars: Optional[Dict[str, Any]] = None,
        **extra_params
    ):
        self.words = words or {}
        self.chars = chars or {}

        for k, v in extra_params.items():
            self.__setattr__(k, v)

    @classmethod
    def from_params(
        cls: Type["FeaturesConfiguration"], params: Params, **extras
    ) -> "FeaturesConfiguration":
        return cls(**params.as_dict(), **extras)

    def compile(self) -> InputFeaturizer:
        """Build input featurizer from configuration"""
        return InputFeaturizer.from_params(Params(copy.deepcopy(vars(self))))


class TokenizerConfiguration(FromParams):
    """"Tokenization configuration"""

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
    """Pipeline configuration attributes"""

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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tokenizer": vars(self.tokenizer),
            "features": vars(self.features),
            "encoder": self.encoder.config,
            "head": self.head.config,
        }


class TrainerConfiguration:
    """Trainer configuration"""

    def __init__(
        self,
        optimizer: Dict[str, Any],
        validation_metric: str = "-loss",
        patience: Optional[int] = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        cuda_device: int = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[Dict[str, Any]] = None,
        momentum_scheduler: Optional[Dict[str, Any]] = None,
        moving_average: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        cache_instances: bool = True,
        in_memory_batches: int = 2,
        data_bucketing: bool = True,
    ):
        self.optimizer = optimizer
        self.validation_metric = validation_metric
        self.patience = patience
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.cuda_device = cuda_device
        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping
        self.learning_rate_scheduler = learning_rate_scheduler
        self.momentum_scheduler = momentum_scheduler
        self.moving_average = moving_average

        # Data Iteration
        self.batch_size = batch_size or 32
        self.data_bucketing = data_bucketing or True
        self.cache_instances = cache_instances or True
        self.in_memory_batches = in_memory_batches


class VocabularyConfiguration:
    """Configures a ``Vocabulary`` before it gets created from data

    Use this to configure a Vocabulary using specific arguments from `allennlp.data.Vocabulary``

    See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])

    Parameters
    ----------
    from_path: ``Optional[str]``
        If provided, try to load model vocab from specified folder path
    sources: ``List[str]``
        List of datasources path used for vocab creation/extension
    min_count: ``Dict[str, int]``
        See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])
    max_vocab_size: ``Union[int, Dict[str, int]]``
        See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])
    pretrained_files: ``Optional[Dict[str, str]]``
        See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])
    only_include_pretrained_words: ``bool``
        See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])
    tokens_to_add: ``Dict[str, List[str]]``
        See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])
    min_pretrained_embeddings: ``Dict[str, int]``
        See [AllenNLP Vocabulary docs](https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary])
    """

    def __init__(
        self,
        from_path: Optional[str] = None,
        sources: List[str] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
    ):
        self.from_path = from_path
        self.sources = sources or []
        self.pretrained_files = pretrained_files
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.only_include_pretrained_words = only_include_pretrained_words
        self.tokens_to_add = tokens_to_add
        self.min_pretrained_embeddings = min_pretrained_embeddings
