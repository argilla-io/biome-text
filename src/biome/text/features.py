from typing import Any, Dict, Optional

from biome.text.modules.configuration import Seq2VecEncoderConfiguration


class WordFeatures:
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
                **({"pretrained_file": self.weights_file} if self.weights_file else {}),
            },
        }

        for k in self.extra_params:
            config[k] = {**self.extra_params[k], **config.get(k)}

        return config

    def to_json(self) -> Dict:
        """Returns the config as dict for the serialized json config file"""
        data = vars(self)
        data.update(data.pop("extra_params"))

        return data

    def to_dict(self) -> Dict:
        """Returns the config as dict"""
        return {
            "embedding_dim": self.embedding_dim,
            "lowercase_tokens": self.lowercase_tokens,
            "trainable": self.trainable,
            "weights_file": self.weights_file,
            **self.extra_params,
        }


class CharFeatures:
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
        data = vars(self)
        data.update(data.pop("extra_params"))

        return data

    def to_dict(self):
        """Returns the config as dict"""
        return {
            "embedding_dim": self.embedding_dim,
            "encoder": self.encoder,
            "dropout": self.dropout,
            **self.extra_params,
        }
