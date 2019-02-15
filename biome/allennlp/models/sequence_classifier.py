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
from torch.nn.functional import softmax

logger = logging.getLogger(__name__)


@Model.register("sequence_classifier")
class SequenceClassifier(Model):
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
        initializer: InitializerApplicator = None,
        regularizer: RegularizerApplicator = None,
        accuracy: Metric = None,
        **kwargs
    ) -> None:
        super(SequenceClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = initializer or InitializerApplicator()
        self._accuracy = accuracy or CategoricalAccuracy()

        self.output_layer = Linear(self.decoder.get_output_dim(), self.num_classes)

        self.__check_configuration()
        self.metrics = {
            label: F1Measure(index)
            for index, label in self.vocab.get_index_to_token_vocabulary(
                "labels"
            ).items()
        }
        self._loss = torch.nn.CrossEntropyLoss()

        self.initializer(self)

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

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict["class_probabilities"]
        if not isinstance(all_predictions, numpy.ndarray):
            all_predictions = all_predictions.data.numpy()

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        for i, probs in enumerate(all_predictions):
            argmax_i = numpy.argmax(probs)
            label = self.vocab.get_token_from_index(argmax_i, namespace="labels")
            label_prob = 0.0

            output_map_probs.append({})
            for j, prob in enumerate(probs):
                label_key = self.vocab.get_token_from_index(j, namespace="labels")
                output_map_probs[i][label_key] = prob
                if label_key == label:
                    label_prob = prob

            max_classes.append(label)
            max_classes_prob.append(label_prob)

        return {
            "logits": output_dict.get("logits"),
            "classes": output_map_probs,
            "max_class": max_classes,
            "max_class_prob": max_classes_prob,
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Get the metrics of our classifier, see :method:`~allennlp.models.Model.get_metrics`.

        Parameter
        ---------
        reset
            Reset the metrics after obtaining them?

        Returns
        -------
        all_metrics
            A dictionary with metric name and value.
        """
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self.metrics.items():
            precision, recall, f1 = metric.get_metric(
                reset
            )  # pylint: disable=invalid-name
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            all_metrics[metric_name + "/f1"] = f1
            all_metrics[metric_name + "/precision"] = precision
            all_metrics[metric_name + "/recall"] = recall

        num_metrics = len(self.metrics)
        all_metrics["average/f1"] = total_f1 / num_metrics
        all_metrics["average/precision"] = total_precision / num_metrics
        all_metrics["average/recall"] = total_recall / num_metrics
        all_metrics["accuracy"] = self._accuracy.get_metric(reset)

        return all_metrics

    def get_class_probabilities(self, logits : torch.Tensor) -> torch.Tensor:
        """Get class probabilities by applying a softmax function.

        Parameter
        ---------
        logits
            The logits of our classification model

        Returns
        -------
        probabilities
            The logits are transformed to probabilities by applying a softmax
        """
        logits_dim = logits.dim()
        # TODO: Describe what we are doing in the following ...
        # From torch.nn.functional.softmax package
        if logits_dim in [0, 1, 3]:
            dim = 0
        else:
            dim = 1
        return softmax(logits, dim=dim)

    def __check_configuration(self):
        """Some basic checks of the architecture."""
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
