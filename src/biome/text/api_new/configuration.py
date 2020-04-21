import copy
from typing import Any, Dict, Optional, Type, Union

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
            "encoder": self.encoder._config,
            "head": self.head._config,
        }
