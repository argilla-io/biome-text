from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

import numpy
import torch
from allennlp.data import Batch
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.data.fields import TextField
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask
from captum.attr import IntegratedGradients

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration

from ..task_head import TaskOutput
from .classification import ClassificationHead


class TextClassification(ClassificationHead):
    """
    Task head for text classification
    """

    label_name = "label"
    forward_arg_name = "text"

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        pooler: Optional[Seq2VecEncoderConfiguration] = None,
        feedforward: Optional[FeedForwardConfiguration] = None,
        multilabel: bool = False,
    ) -> None:

        super(TextClassification, self).__init__(backbone, labels, multilabel)

        self.pooler = (
            pooler.input_dim(self.backbone.encoder.get_output_dim()).compile()
            if pooler
            else BagOfEmbeddingsEncoder(self.backbone.encoder.get_output_dim())
        )
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
        instance = self.backbone.featurizer(
            text,
            to_field=self.forward_arg_name,
            aggregate=True,
            exclude_record_keys=True,
        )
        return self.add_label(instance, label, to_field=self.label_name)

    def forward(  # type: ignore
        self,
        text: TextFieldTensors,
        label: torch.IntTensor = None,
    ) -> TaskOutput:

        mask = get_text_field_mask(text)
        embedded_text = self.backbone.forward(text, mask)
        embedded_text = self.pooler(embedded_text, mask=mask)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return self.calculate_output(logits=logits, label=label)

    def explain_prediction(
        self, prediction: Dict[str, numpy.array], instance: Instance, n_steps: int
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
            self.backbone.vocab, prediction["labels"][0]
        )
        attributions, delta = ig.attribute(
            text_embeddings,
            n_steps=n_steps,
            target=label_id,
            additional_forward_args=mask,
            return_convergence_delta=True,
        )
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
        embedded_text = self.pooler.forward(embedded_text, mask)
        if self.feedforward:
            embedded_text = self.feedforward.forward(embedded_text)
        return self._classification_layer(embedded_text)


class TextClassificationConfiguration(ComponentConfiguration[TextClassification]):
    """Configuration for classification head components"""

    pass
