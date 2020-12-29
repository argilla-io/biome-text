from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional

from biome.text.modules.configuration import Seq2VecEncoderConfiguration


class Features(ABC):
    """All features used in the pipeline configuration inherit from this abstract class."""

    @property
    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def to_json(self):
        pass

    def __eq__(self, other):
        return all([a == b for a, b in zip(vars(self), vars(other))])


class WordFeatures(Features):
    """Feature configuration at word level

    Parameters
    ----------
    embedding_dim
        Dimension of the embeddings
    lowercase_tokens
        If True, lowercase tokens before the indexing
    trainable
        If False, freeze the embeddings
    weights_file
        Path to a file with pretrained weights for the embedding
    **extra_params
        Extra parameters passed on to the `indexer` and `embedder` of the AllenNLP configuration framework.
        For example: `WordFeatures(embedding_dim=300, embedder={"padding_index": 0})`
    """

    namespace = "word"

    def __init__(
        self,
        embedding_dim: int,
        lowercase_tokens: bool = False,
        trainable: bool = True,
        weights_file: Optional[str] = None,
        **extra_params
    ):
        self.embedding_dim = embedding_dim
        self.lowercase_tokens = lowercase_tokens
        self.trainable = trainable
        self.weights_file = weights_file
        self.extra_params = extra_params

    @property
    def config(self) -> Dict:
        """Returns the config in AllenNLP format"""
        config = {
            "indexer": {
                "type": "single_id",
                "lowercase_tokens": self.lowercase_tokens,
                "namespace": self.namespace,
            },
            "embedder": {
                "embedding_dim": self.embedding_dim,
                "vocab_namespace": self.namespace,
                "trainable": self.trainable,
                "pretrained_file": self.weights_file,
            },
        }

        for k in self.extra_params:
            config[k] = {**self.extra_params[k], **config.get(k)}

        return config

    def to_json(self) -> Dict:
        """Returns the config as dict for the serialized json config file"""
        return {
            "embedding_dim": self.embedding_dim,
            "lowercase_tokens": self.lowercase_tokens,
            "trainable": self.trainable,
            "weights_file": self.weights_file,
            **self.extra_params,
        }


class CharFeatures(Features):
    """Feature configuration at character level

    Parameters
    ----------
    embedding_dim
        Dimension of the character embeddings.
    encoder
        A sequence to vector encoder resulting in a word representation based on its characters
    dropout
        Dropout applied to the output of the encoder
    lowercase_characters
        If True, lowercase characters before the indexing
    **extra_params
        Extra parameters passed on to the `indexer` and `embedder` of the AllenNLP configuration framework.
        For example: `CharFeatures(embedding_dim=32, indexer={"min_padding_length": 5}, ...)`
    """

    namespace = "char"

    def __init__(
        self,
        embedding_dim: int,
        encoder: Dict[str, Any],
        dropout: float = 0.0,
        lowercase_characters: bool = False,
        **extra_params
    ):
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.dropout = dropout
        self.lowercase_characters = lowercase_characters
        self.extra_params = extra_params

    @property
    def config(self) -> Dict:
        """Returns the config in AllenNLP format"""
        config = {
            "indexer": {
                "type": "characters",
                "namespace": self.namespace,
                "character_tokenizer": {
                    "lowercase_characters": self.lowercase_characters
                },
            },
            #         "character_tokenizer": {"lowercase_characters": True},
            "embedder": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": self.embedding_dim,
                    "vocab_namespace": self.namespace,
                },
                "encoder": Seq2VecEncoderConfiguration(**self.encoder)
                .input_dim(self.embedding_dim)
                .config,
                "dropout": self.dropout,
            },
        }

        for k, v in self.extra_params.items():
            config[k] = {**self.extra_params[k], **config.get(k)}

        return config

    def to_json(self):
        """Returns the config as dict for the serialized json config file"""
        return {
            "embedding_dim": self.embedding_dim,
            "lowercase_characters": self.lowercase_characters,
            "encoder": self.encoder,
            "dropout": self.dropout,
            **self.extra_params,
        }


class TransformersFeatures(Features):
    """Configuration of the feature extracted with the [transformers models](https://huggingface.co/models).

    We use AllenNLPs "mismatched" indexer and embedder to get word-level representations.
    Most of the transformers models work with word-piece tokenizers.

    Parameters
    ----------
    model_name
        Name of one of the [transformers models](https://huggingface.co/models).
    trainable
        If false, freeze the transformer weights
    max_length
        If positive, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation.
    last_layer_only
        When `True`, only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.

    Attributes
    ----------
    mismatched: bool (default: True)
        If true, we will sum up the word-piece vectors in a word to pull out a single vector for each word.
        If you use a transformers tokenizer, this is automatically set to False by the `PipelineConfiguration`.
    """

    namespace = "transformers"

    def __init__(
        self,
        model_name: str,
        trainable: bool = False,
        max_length: Optional[int] = None,
        last_layer_only: bool = True,
    ):
        self.model_name = model_name
        self.trainable = trainable
        self.max_length = max_length
        self.last_layer_only = last_layer_only
        self.mismatched = True

    @property
    def config(self) -> Dict:
        """Returns the config in AllenNLP format"""
        config = {
            "indexer": {
                "type": "pretrained_transformer_mismatched"
                if self.mismatched
                else "pretrained_transformer",
                "model_name": self.model_name,
                "namespace": self.namespace,
                "max_length": self.max_length,
            },
            "embedder": {
                "type": "pretrained_transformer_mismatched"
                if self.mismatched
                else "pretrained_transformer",
                "model_name": self.model_name,
                "train_parameters": self.trainable,
                "max_length": self.max_length,
                "last_layer_only": self.last_layer_only,
            },
        }

        return config

    def to_json(self) -> Dict:
        """Returns the config as dict for the serialized json config file"""
        return {
            "model_name": self.model_name,
            "trainable": self.trainable,
            "max_length": self.max_length,
            "last_layer_only": self.last_layer_only,
        }
