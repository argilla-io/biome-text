from typing import Any, Dict, Optional

from biome.text.modules.configuration import Seq2VecEncoderConfiguration


class WordFeatures:
    """Feature configuration at word level"""

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
    def config(self):
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

    def to_json(self):
        data = vars(self)
        data.update(data.pop("extra_params"))

        return data

    def to_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "lowercase_tokens": self.lowercase_tokens,
            "trainable": self.trainable,
            "weights_file": self.weights_file,
            **self.extra_params,
        }


class CharFeatures:
    """Feature configuration at character level"""

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
    def config(self):
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
        data = vars(self)
        data.update(data.pop("extra_params"))

        return data

    def to_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "encoder": self.encoder,
            "dropout": self.dropout,
            **self.extra_params,
        }
