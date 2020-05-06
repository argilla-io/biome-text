import copy
from typing import Any, Dict, Optional

from allennlp.common import Params
from allennlp.data import TokenIndexer, Vocabulary
from allennlp.modules import TextFieldEmbedder

from biome.text.api_new.commons import InmutableDict
from biome.text.api_new.modules.specs import Seq2VecEncoderSpec

Embedder = TextFieldEmbedder


class _WordFeaturesSpecs:
    """Simplifies word level features configuration """

    namespace = "words"

    def __init__(
        self,
        embedding_dim: int,
        lowercase_tokens: bool = False,
        trainable: bool = True,
        weights_file: Optional[str] = None,
        **extra_params
    ):
        self.config = {
            "indexer": {
                "type": "single_id",
                "lowercase_tokens": lowercase_tokens,
                "namespace": self.namespace,
            },
            "embedder": {
                "embedding_dim": embedding_dim,
                "vocab_namespace": self.namespace,
                "trainable": trainable,
                **({"pretrained_file": weights_file} if weights_file else {}),
            },
        }

        for k, v in extra_params.items():
            self.config[k] = {**extra_params[k], **self.config.get(k)}


class _CharacterFeaturesSpec:
    """Simplifies character level features configuration"""

    namespace = "chars"

    def __init__(
        self,
        embedding_dim: int,
        encoder: Dict[str, Any],
        dropout: int = 0.0,
        **extra_params
    ):
        self.config = {
            "indexer": {"type": "characters", "namespace": self.namespace},
            "embedder": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": embedding_dim,
                    "vocab_namespace": self.namespace,
                },
                "encoder": Seq2VecEncoderSpec(**encoder)
                .input_dim(embedding_dim)
                .config,
                "dropout": dropout,
            },
        }

        for k, v in extra_params.items():
            self.config[k] = {**extra_params[k], **self.config.get(k)}


class InputFeaturizer:
    """Transforms input text (words and/or characters) into indexes and embedding vectors.

    This class defines two input features, words and chars for embeddings at word and character level respectively.

    You can provide additional features by manually specify `indexer` and `embedder` configurations within each
    input feature.

    Parameters
    ----------
    words : ``Dict[str, Any]``
        Dictionary defining how to index and embed words
    chars : ``Dict[str, Any]``
        Dictionary defining how to encode and embed characters
    kwargs :
        Additional params for setting up the features
    """

    __DEFAULT_CONFIG = {"embedding_dim": 50}
    __INDEXER_KEYNAME = "indexer"
    __EMBEDDER_KEYNAME = "embedder"

    WORDS = _WordFeaturesSpecs.namespace
    CHARS = _CharacterFeaturesSpec.namespace

    def __init__(
        self,
        words: Optional[Dict[str, Any]] = None,
        chars: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Dict[str, Any]]
    ):

        configuration = kwargs or {}

        if not (words or chars or configuration):
            words = self.__DEFAULT_CONFIG

        if words:
            self.words = _WordFeaturesSpecs(**words).config

        if chars:
            self.chars = _CharacterFeaturesSpec(**chars).config

        for k, v in configuration.items():
            self.__setattr__(k, v)

    @classmethod
    def from_params(cls, params: Params) -> "InputFeaturizer":
        """ Loads featurizer from ``allennlp.Params``
        
        Parameters
        ----------
        params : ``Params``
            Params for the featurizer configuration

        Returns
        -------
        An instance of ``InputFeaturizer``
        """
        return cls(**params.as_dict())

    @property
    def config(self):
        """The featurizer configuration"""
        return copy.deepcopy(self.__dict__)

    @property
    def feature_keys(self):
        """The configured feature names ("words", "chars", ...)"""
        return list(self.__dict__.keys())

    def build_features(self) -> Dict[str, TokenIndexer]:
        """Builds configured token indexers features as allennlp token indexers.
        
        Returns
        -------
        An `InmutableDict` defining the token indexers of the featurizer
        """
        # fmt: off
        return InmutableDict({
            feature: TokenIndexer.from_params(Params(config[self.__INDEXER_KEYNAME]))
            for feature, config in self.config.items()
        })
        # fmt: on

    def build_embedder(self, vocab: Vocabulary) -> Embedder:
        """Builds a `TextFieldEmbedder` from configured embedding features
        
        Parameters
        ----------
        vocab : ``Vocabulary``
            Vocabulary object to be used by the embedding layers
        Returns
        -------
        A `TextFieldEmbedder`
        """
        # fmt: off
        return TextFieldEmbedder.from_params(
            Params({
                feature: config[self.__EMBEDDER_KEYNAME]
                for feature, config in self.config.items()}
            ),
            vocab=vocab
        )
        # fmt: on

    def __setattr__(self, key, value):
        self.__dict__[key] = value
