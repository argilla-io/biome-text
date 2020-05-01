from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, BiMpmMatching

from .defs import ComponentSpec


class Seq2VecEncoderSpec(ComponentSpec[Seq2VecEncoder]):
    """Layer spec for Seq2VecEncoder components"""

    pass


class Seq2SeqEncoderSpec(ComponentSpec[Seq2SeqEncoder]):
    """Layer spec for Seq2SeqEncoder components"""

    pass


class FeedForwardSpec(ComponentSpec[FeedForward]):
    """Layer spec for FeedForward components"""

    pass


class BiMpmMatchingSpec(ComponentSpec[BiMpmMatching]):
    """Layer spec for BiMpmMatching components"""

    pass
