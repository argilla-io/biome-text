from allennlp.modules import BiMpmMatching
from allennlp.modules import Embedding
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import Seq2VecEncoder

from .defs import ComponentConfiguration


class Seq2VecEncoderConfiguration(ComponentConfiguration[Seq2VecEncoder]):
    """Layer spec for Seq2VecEncoder components"""

    pass


class Seq2SeqEncoderConfiguration(ComponentConfiguration[Seq2SeqEncoder]):
    """Layer spec for Seq2SeqEncoder components"""

    pass


class FeedForwardConfiguration(ComponentConfiguration[FeedForward]):
    """Layer spec for FeedForward components"""

    pass


class BiMpmMatchingConfiguration(ComponentConfiguration[BiMpmMatching]):
    """Layer spec for BiMpmMatching components"""

    pass


class EmbeddingConfiguration(ComponentConfiguration[Embedding]):
    """Layer spec for Embedding components"""

    pass
