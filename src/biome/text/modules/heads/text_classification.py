from typing import Any, Dict, List, Optional, Union, cast

import numpy
import torch
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.nn.util import get_text_field_mask
from captum.attr import IntegratedGradients

from biome.text.backbone import ModelBackbone
from biome.text.modules.specs import (
    ComponentSpec,
    FeedForwardSpec,
    Seq2VecEncoderSpec,
)
from biome.text import vocabulary
from .classification.defs import ClassificationHead
from .defs import TaskOutput


class TextClassification(ClassificationHead):
    """
    Task head for text classification
    """

    label_name = "label"
    forward_arg_name = "text"

    def __init__(
        self,
        backbone: ModelBackbone,
        pooler: Seq2VecEncoderSpec,
        labels: List[str],
        feedforward: Optional[FeedForwardSpec] = None,
        multilabel: bool = False,
    ) -> None:

        super(TextClassification, self).__init__(backbone, labels, multilabel)

        self.pooler = pooler.input_dim(self.backbone.encoder.get_output_dim()).compile()
        self.feedforward = (
            None
            if not feedforward
            else feedforward.input_dim(self.pooler.get_output_dim()).compile()
        )
        self._classification_layer = torch.nn.Linear(
            (self.feedforward or self.pooler).get_output_dim(), self.num_labels
        )

    def featurize(
        self, text: Any, label: Optional[Union[int, str, List[Union[int, str]]]] = None
    ) -> Optional[Instance]:
        instance = self.backbone.featurize(
            text, to_field=self.forward_arg_name, aggregate=True
        )
        return self.add_label(instance, label, to_field=self.label_name)

    def forward(  # type: ignore
        self, text: Dict[str, torch.Tensor], label: torch.IntTensor = None,
    ) -> TaskOutput:

        mask = get_text_field_mask(text)
        embedded_text = self.backbone.forward(text, mask)
        embedded_text = self.pooler(embedded_text, mask=mask)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return self.calculate_output(logits=logits, label=label)

    def explain_prediction(
        self, prediction: Dict[str, numpy.array], instance: Instance
    ) -> Dict[str, Any]:

        dataset = Batch([instance])
        input_tokens_ids = dataset.as_tensor_dict()
        ig = IntegratedGradients(self._explain_embeddings)

        num_wrapping_dims = 0

        text_tokens = [
            token.text
            for token in cast(TextField, instance.get(self.forward_arg_name)).tokens
        ]
        text_tensor = input_tokens_ids.get(self.forward_arg_name)
        mask = get_text_field_mask(text_tensor, num_wrapping_dims=num_wrapping_dims)
        text_embeddings = self.backbone.embedder.forward(
            text_tensor, num_wrapping_dims=num_wrapping_dims
        )

        label_id = vocabulary.index_for_label(
            self.backbone.vocab, prediction.get(self.label_name)
        )
        attributions, delta = ig.attribute(
            text_embeddings,
            target=label_id,
            additional_forward_args=mask,
            return_convergence_delta=True,
        )
        # TODO: what the attribution and deltas means
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().numpy()

        return {
            **prediction,
            "explain": {
                self.forward_arg_name: [
                    {"token": token, "attribution": attribution}
                    for token, attribution in zip(text_tokens, attributions)
                ]
            },
        }

    def _explain_embeddings(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Explain embeddings for single text classification task"""
        embedded_text = self.backbone.encoder.forward(embeddings, mask)
        embedded_text = self.pooler.forward(embedded_text)
        if self.feedforward:
            embedded_text = self.feedforward.forward(embedded_text)
        logits = self._classification_layer(embedded_text)
        # TODO: review what kind of information we need to pass
        return logits


class TextClassificationSpec(ComponentSpec[TextClassification]):
    """Spec for classification head components"""

    pass
