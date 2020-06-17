from allennlp.modules import BiMpmMatching, FeedForward, Seq2SeqEncoder, Seq2VecEncoder

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
