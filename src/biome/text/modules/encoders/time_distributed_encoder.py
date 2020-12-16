from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import TimeDistributed


class TimeDistributedEncoder(Seq2SeqEncoder):
    """Wraps a Seq2SeqEncoder into a TimeDistributed module and implements the Seq2SeqEncoder API"""

    def __init__(self, encoder: Seq2SeqEncoder):
        super(TimeDistributedEncoder, self).__init__()

        self._input_dim = encoder.get_input_dim()
        self._output_dim = encoder.get_output_dim()
        self._is_bidirectional = (
            hasattr(encoder, "is_bidirectional") and encoder.is_bidirectional()
        )

        self._encoder = TimeDistributed(encoder)

    def forward(self, *input, **inputs):
        return self._encoder(*input, **inputs)

    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    def get_output_dim(self) -> int:
        return self._output_dim

    def get_input_dim(self):
        return self._input_dim
