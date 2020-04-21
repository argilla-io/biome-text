from typing import Any, Dict, List, Optional, Union, cast

import numpy
import torch
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import ListField, TextField
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask
from captum.attr import IntegratedGradients

from biome.text.api_new.modules.specs import ComponentSpec
from biome.text.api_new.modules.specs import (
    FeedForwardSpec,
    Seq2SeqEncoderSpec,
    Seq2VecEncoderSpec,
)
from .classification.defs import ClassificationHead
from .defs import TaskOutput
from biome.text.api_new.modules.encoders import TimeDistributedEncoder
from biome.text.api_new.model import Model
from biome.text.api_new.vocabulary import vocabulary


class DocumentClassification(ClassificationHead):
    """
    Task head for document text classification. It's quite similar to text
    classification but including the doc2vec transformation layers
    """

    forward_arg_name = "document"
    label_name = "label"

    def __init__(
        self,
        model: Model,
        pooler: Seq2VecEncoderSpec,
        labels: List[str],
        tokens_pooler: Optional[Seq2VecEncoderSpec] = None,
        encoder: Optional[Seq2SeqEncoderSpec] = None,
        feedforward: Optional[FeedForwardSpec] = None,
        multilabel: bool = False,
    ) -> None:

        super(DocumentClassification, self).__init__(
            model, labels=labels, multilabel=multilabel
        )

        self.model.encoder = TimeDistributedEncoder(model.encoder)

        # layers
        self.tokens_pooler = TimeDistributedEncoder(
            BagOfEmbeddingsEncoder(embedding_dim=self.model.encoder.get_output_dim())
            if not tokens_pooler
            else tokens_pooler.input_dim(self.model.encoder.get_output_dim()).compile()
        )
        self.encoder = (
            PassThroughEncoder(self.tokens_pooler.get_output_dim())
            if not encoder
            else encoder.input_dim(self.tokens_pooler.get_output_dim()).compile()
        )
        self.pooler = pooler.input_dim(self.encoder.get_output_dim()).compile()
        self.feedforward = (
            None
            if not feedforward
            else feedforward.input_dim(self.pooler.get_output_dim()).compile()
        )

        self._classification_layer = torch.nn.Linear(
            (self.feedforward or self.pooler).get_output_dim(), self.num_labels
        )

    def featurize(
        self,
        document: Any,
        label: Optional[Union[int, str, List[Union[int, str]]]] = None,
    ) -> Optional[Instance]:
        instance = self.model.featurize(document, to_field=self.forward_arg_name)
        return self.add_label(instance, label, to_field=self.label_name)

    def forward(
        self, document: Dict[str, torch.Tensor], label: torch.IntTensor = None
    ) -> TaskOutput:
        mask = get_text_field_mask(
            document, num_wrapping_dims=1
        )  # Why num_wrapping_dims=1 !?
        embedded_text = self.model.forward(document, mask, num_wrapping_dims=1)
        embedded_text = self.tokens_pooler(embedded_text, mask=mask)

        mask = get_text_field_mask(
            {self.forward_arg_name: embedded_text}
        )  # Add an extra dimension to tensor mask
        embedded_text = self.encoder(embedded_text, mask=mask)
        embedded_text = self.pooler(embedded_text, mask=mask)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return self.calculate_output(logits=logits, label=label)

    def prediction_explain(
        self, prediction: Dict[str, numpy.array], instance: Instance
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
        text_embeddings = self.model.embedder.forward(
            document_tensors, num_wrapping_dims=num_wrapping_dims
        )

        label_id = vocabulary.index_for_label(
            self.model.vocab, prediction.get(self.label_name)
        )
        attributions, delta = ig.attribute(
            text_embeddings,
            target=label_id,
            additional_forward_args=mask,
            return_convergence_delta=True,
        )
        # TODO: what the attribution and deltas means
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
        embedded_text = self.model.encoder.forward(embeddings, mask)
        embedded_text = self.tokens_pooler(embedded_text, mask=mask)

        mask = get_text_field_mask(
            {self.forward_arg_name: embedded_text}
        )  # Add an extra dimension to tensor mask
        embedded_text = self.encoder(embedded_text, mask=mask)
        embedded_text = self.pooler(embedded_text, mask=mask)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return self.calculate_output(logits=logits, label=None).probs


class DocumentClassificationSpec(ComponentSpec[DocumentClassification]):
    """Lazy initialization for document classification head components"""

    pass
