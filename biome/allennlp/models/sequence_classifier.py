import logging
from typing import Dict, Optional

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Metric
from overrides import overrides

from torch.nn.modules.linear import Linear

from biome.allennlp.models import AbstractClassifier

logger = logging.getLogger(__name__)


@Model.register("sequence_classifier")
class SequenceClassifier(AbstractClassifier):
    """
    This ``SequenceClassifier`` simply encodes a sequence of text with a ``Seq2VecEncoder``, then
    predicts a label for the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    decoder : ``Feedforward``
        The decoder that will decode the final answer of the model
    pre_encoder : ``allennlp.modules.FeedForward``, optional (default = None)
        Feedforward layer to be applied to embedded tokens.
    encoder : ``Seq2VecEncoder``, optional (default = None)
        The encoder  that we will use in between embedding tokens
        and predicting output tags.
    initializer : ``InitializerApplicator``, optional (default = ``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default = None)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        decoder: FeedForward,
        encoder: Seq2VecEncoder = None,
        pre_encoder: FeedForward = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: RegularizerApplicator = None,
        accuracy: Metric = None,
    ) -> None:
        super(SequenceClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = Linear(self.decoder.get_output_dim(), self.num_classes)

        self.__check_configuration()
        self._accuracy = accuracy or CategoricalAccuracy()
        self.metrics = {
            label: F1Measure(index)
            for index, label in self.vocab.get_index_to_token_vocabulary(
                "labels"
            ).items()
        }
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.Tensor],
        gold_label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        gold_label : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class label of shape
            ``(batch_size, num_classes)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        if self.pre_encoder:
            embedded_text_input = self.pre_encoder(embedded_text_input)
        if self.encoder:
            encoded_text = self.encoder(embedded_text_input, mask)
        else:
            encoded_text = embedded_text_input[:, 0]  # Bert

        decoded_text = self.decoder(encoded_text)

        logits = self.output_layer(decoded_text)

        class_probabilities = self.get_class_probabilities(logits)
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if gold_label is not None:
            loss = self._loss(logits, gold_label.long())
            output_dict["loss"] = loss
            self._accuracy(logits, gold_label)
            for name, metric in self.metrics.items():
                metric(logits, gold_label)

        return output_dict

    def __check_configuration(self):
        encoder = self.encoder
        if encoder:
            if self.pre_encoder:
                encoder = self.pre_encoder
            if self.text_field_embedder.get_output_dim() != encoder.get_input_dim():
                raise ConfigurationError(
                    "The output dimension of the text_field_embedder must match the "
                    "input dimension of the sequence encoder. Found {} and {}, "
                    "respectively.".format(
                        self.text_field_embedder.get_output_dim(),
                        encoder.get_input_dim(),
                    )
                )
        else:
            # TODO: Add more checks
            pass
