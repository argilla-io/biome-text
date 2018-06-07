from typing import Dict, Optional
import logging

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from recognai.models.archival import load_archive

_logger = logging.getLogger(__name__)


@Model.register("abstract_classifier")
class AbstractClassifier(Model):
    """
    This ``AbstractClassifier`` simply encodes a sequence of text with a ``Seq2VecEncoder``, then
    predicts a label for the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
    """

    def __init__(self, vocab: Vocabulary,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AbstractClassifier, self).__init__(vocab, regularizer)

        self._accuracy = CategoricalAccuracy()
        self.metrics = {label: F1Measure(index) for index, label
                        in self.vocab.get_index_to_token_vocabulary("labels").items()}
        self._loss = torch.nn.CrossEntropyLoss()

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                gold_label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

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
        encoded_text = self.encoder(embedded_text_input, mask)
        logits = self.projection_layer(encoded_text)

        class_probabilities = F.softmax(logits)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if gold_label is not None:
            loss = self._loss(logits, gold_label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, gold_label.squeeze(-1))
            for name, metric in self.metrics.items():
                metric(logits, gold_label.squeeze(-1))

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        if not isinstance(all_predictions, numpy.ndarray):
            all_predictions = all_predictions.data.numpy()

        output_map_probs = []
        max_labels = []
        for i, probs in enumerate(all_predictions):
            argmax_i = numpy.argmax(probs)
            label = self.vocab.get_token_from_index(argmax_i, namespace="labels")
            max_labels.append(label)

            output_map_probs.append({})
            for j, prob in enumerate(probs):
                label_key = self.vocab.get_token_from_index(j, namespace="labels")
                output_map_probs[i][label_key] = prob

        output_dict['probabilities_by_class'] = output_map_probs
        output_dict['max_label'] = max_labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self.metrics.items():
            precision, recall, f1 = metric.get_metric(reset)  # pylint: disable=invalid-name
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            all_metrics[metric_name + "_f1"] = f1
            all_metrics[metric_name + "_precision"] = precision
            all_metrics[metric_name + "_recall"] = recall

        num_metrics = len(self.metrics)
        all_metrics["average_f1"] = total_f1 / num_metrics
        all_metrics["average_precision"] = total_precision / num_metrics
        all_metrics["average_recall"] = total_recall / num_metrics
        all_metrics['accuracy'] = self._accuracy.get_metric(reset)

        return all_metrics

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AbstractClassifier':
        embedder_params = params.pop("text_field_embedder")
        # text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        # encoder_params = params.pop("encoder", None)
        # if encoder_params is not None:
        #     encoder = Seq2VecEncoder.from_params(encoder_params)
        # else:
        #     encoder = None

        # initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   regularizer=regularizer)

    @classmethod
    def try_from_location(cls, params: Params) -> 'AbstractClassifier':
        model_location = params.get('model_location', None)
        if model_location:
            try:
                archive = load_archive(archive_file=model_location)
                _logger.warning("Loaded model from location %s", model_location)
                return archive.model
            except:
                _logger.warning("Cannot load model from location %s", model_location)
                return None
