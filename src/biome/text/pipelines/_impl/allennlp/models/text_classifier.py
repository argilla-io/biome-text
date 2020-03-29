from typing import Dict, Optional, Any, List

import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from overrides import overrides

from biome.text.pipelines._impl.allennlp.models.defs import ITextClassifier
from biome.text.pipelines._impl.allennlp.modules.heads.classification_task_header import (
    ClassificationHead,
)

try:
    from allennlp.data import TextFieldTensors
except ImportError:
    TextFieldTensors = Dict[str, torch.Tensor]


class TextClassifier(ITextClassifier):
    """
    # Parameters
    vocab : `Vocabulary`
    embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    encoder : `Seq2SeqEncoder` (default=`PassThroughEncoder`)
        Optional Seq2Seq encoder layer for the input text.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        head: ClassificationHead,
        encoder: Optional[Seq2SeqEncoder] = None,
        initializer: Optional[InitializerApplicator] = None,
    ) -> None:

        super(TextClassifier, self).__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder or PassThroughEncoder(embedder.get_output_dim())
        self._head = head
        self._label_namespace = "labels"

        self._num_wrapping_dims = 1
        self._encoder = TimeDistributed(encoder)

        (initializer or InitializerApplicator())(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        embedded_text = self._embedder(
            tokens, num_wrapping_dims=self._num_wrapping_dims
        )
        mask = get_text_field_mask(tokens, num_wrapping_dims=self._num_wrapping_dims)
        embedded_text = self._encoder(embedded_text, mask=mask)
        output_dict = self._head(embedded_text, mask=mask, label=label)

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.make_output_human_readable(output_dict)

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        all_predictions = output_dict["probs"]
        if not isinstance(all_predictions, np.ndarray):
            all_predictions = all_predictions.data.numpy()

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        for i, probs in enumerate(all_predictions):
            argmax_i = np.argmax(probs)
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

        return_dict = {
            "logits": output_dict.get("logits"),
            "classes": output_map_probs,
            "max_class": max_classes,
            "max_class_prob": max_classes_prob,
        }
        # having loss == None in dict (when no label is present) fails
        if "loss" in output_dict.keys():
            return_dict["loss"] = output_dict.get("loss")
        return return_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._head.get_metrics(reset)

    def output_classes(self) -> List[str]:
        return self._head.output_classes

    def extend_labels(self, labels: List[str]):
        self._head.extend_labels(labels)
