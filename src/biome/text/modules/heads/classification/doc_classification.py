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
from allennlp.data.fields import ListField
from allennlp.data.fields import TextField
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask
from captum.attr import IntegratedGradients

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2SeqEncoderConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.encoders import TimeDistributedEncoder

from ..task_head import TaskOutput
from .classification import ClassificationHead


class DocumentClassification(ClassificationHead):
    """
    Task head for document text classification. It's quite similar to text
    classification but including the doc2vec transformation layers
    """

    forward_arg_name = "document"
    label_name = "label"

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        tokens_pooler: Optional[Seq2VecEncoderConfiguration] = None,
        sentences_encoder: Optional[Seq2SeqEncoderConfiguration] = None,
        sentences_pooler: Seq2VecEncoderConfiguration = None,
        feedforward: Optional[FeedForwardConfiguration] = None,
        multilabel: bool = False,
    ) -> None:

        super(DocumentClassification, self).__init__(
            backbone, labels=labels, multilabel=multilabel
        )

        self.backbone.encoder = TimeDistributedEncoder(backbone.encoder)

        # layers
        self.tokens_pooler = TimeDistributedEncoder(
            BagOfEmbeddingsEncoder(embedding_dim=self.backbone.encoder.get_output_dim())
            if not tokens_pooler
            else tokens_pooler.input_dim(
                self.backbone.encoder.get_output_dim()
            ).compile()
        )
        self.sentences_encoder = (
            PassThroughEncoder(self.tokens_pooler.get_output_dim())
            if not sentences_encoder
            else sentences_encoder.input_dim(
                self.tokens_pooler.get_output_dim()
            ).compile()
        )
        self.sentences_pooler = (
            BagOfEmbeddingsEncoder(self.sentences_encoder.get_output_dim())
            if not sentences_pooler
            else sentences_pooler.input_dim(
                self.sentences_encoder.get_output_dim()
            ).compile()
        )
        self.feedforward = (
            None
            if not feedforward
            else feedforward.input_dim(self.sentences_pooler.get_output_dim()).compile()
        )

        self._classification_layer = torch.nn.Linear(
            (self.feedforward or self.sentences_pooler).get_output_dim(),
            self.num_labels,
        )

    def featurize(
        self,
        document: Any,
        label: Optional[Union[int, str, List[Union[int, str]]]] = None,
    ) -> Optional[Instance]:
        instance = self.backbone.featurizer(
            document, to_field=self.forward_arg_name, exclude_record_keys=True
        )
        return self.add_label(instance, label, to_field=self.label_name)

    def forward(
        self, document: TextFieldTensors, label: torch.IntTensor = None
    ) -> TaskOutput:
        mask = get_text_field_mask(
            document, num_wrapping_dims=1
        )  # Why num_wrapping_dims=1 !?
        embedded_text = self.backbone.forward(document, mask, num_wrapping_dims=1)
        embedded_text = self.tokens_pooler(embedded_text, mask=mask)

        # Here we need to mask the TextFields that only contain the padding token -> last dimension only contains False
        # Those fields were added to possibly equalize the batch.
        mask = torch.sum(mask, -1) > 0
        embedded_text = self.sentences_encoder(embedded_text, mask=mask)
        embedded_text = self.sentences_pooler(embedded_text, mask=mask)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return self.calculate_output(logits=logits, label=label)

    def explain_prediction(
        self, prediction: Dict[str, numpy.array], instance: Instance, n_steps: int
    ) -> Dict[str, Any]:
        """Here, we must apply transformations for manage ListFields tensors shapes"""

        dataset = Batch([instance])
        input_tokens_ids = dataset.as_tensor_dict()
        ig = IntegratedGradients(self._explain_embeddings)

        num_wrapping_dims = 1

        document_tokens = [
            [token.text for token in cast(TextField, text_field).tokens]
            for text_field in cast(ListField, instance.get(self.forward_arg_name))
        ]
        document_tensors = input_tokens_ids.get(self.forward_arg_name)
        mask = get_text_field_mask(
            document_tensors, num_wrapping_dims=num_wrapping_dims
        )
        text_embeddings = self.backbone.embedder.forward(
            document_tensors, num_wrapping_dims=num_wrapping_dims
        )

        label_id = vocabulary.index_for_label(
            self.backbone.vocab, prediction.get(self.label_name)
        )
        attributions, delta = ig.attribute(
            text_embeddings,
            target=label_id,
            additional_forward_args=mask,
            return_convergence_delta=True,
            n_steps=n_steps,
        )
        attributions = attributions.sum(dim=3).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().numpy()

        return {
            **prediction,
            "explain": {
                self.forward_arg_name: [
                    [
                        {"token": token, "attribution": attribution}
                        for token, attribution in zip(
                            sentence_tokens, sentence_attribution
                        )
                    ]
                    for sentence_tokens, sentence_attribution in zip(
                        document_tokens, attributions
                    )
                ]
            },
        }

    def _explain_embeddings(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Embedding interpret for ListField shapes"""
        embedded_text = self.backbone.encoder.forward(embeddings, mask)
        embedded_text = self.tokens_pooler(embedded_text, mask=mask)

        sentences_mask = torch.sum(mask, -1) > 0
        embedded_text = self.sentences_encoder(embedded_text, mask=sentences_mask)
        embedded_text = self.sentences_pooler(embedded_text, mask=sentences_mask)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return logits


class DocumentClassificationConfiguration(
    ComponentConfiguration[DocumentClassification]
):
    """Lazy initialization for document classification head components"""

    pass
