from typing import Any, Dict, Optional

from biome.text.modules.specs import Seq2VecEncoderSpec


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


class CharFeatures:
    """Feature configuration at character level"""

    namespace = "char"

    def __init__(
        self,
        embedding_dim: int,
        encoder: Dict[str, Any],
        dropout: float = 0.0,
        **extra_params
    ):
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.dropout = dropout
        self.extra_params = extra_params

    @property
    def config(self):
        config = {
            "indexer": {"type": "characters", "namespace": self.namespace},
            "embedder": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": self.embedding_dim,
                    "vocab_namespace": self.namespace,
                },
                "encoder": Seq2VecEncoderSpec(**self.encoder)
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
