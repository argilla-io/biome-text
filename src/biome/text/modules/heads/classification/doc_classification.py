from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

import numpy
import torch
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.data.fields import ListField
from allennlp.data.fields import TextField
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask
from captum.attr import IntegratedGradients

from biome.text.backbone import ModelBackbone
from biome.text.featurizer import FeaturizeError
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2SeqEncoderConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.encoders import TimeDistributedEncoder
from biome.text.modules.heads.classification.classification import ClassificationHead
from biome.text.modules.heads.task_prediction import Attribution
from biome.text.modules.heads.task_prediction import DocumentClassificationPrediction


class DocumentClassification(ClassificationHead):
    """
    Task head for document text classification. It's quite similar to text
    classification but including the doc2vec transformation layers
    """

    forward_arg_name = "text"
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
        text: Union[List[str], Dict[str, str]],
        label: Optional[Union[str, List[str]]] = None,
    ) -> Optional[Instance]:
        try:
            instance = self.backbone.featurizer(
                text, to_field=self.forward_arg_name, exclude_record_keys=True
            )
        except FeaturizeError as error:
            self._LOGGER.exception(error)
            return None

        return self._add_label(instance, label, to_field=self.label_name)

    def forward(
        self, text: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, Any]:
        mask = get_text_field_mask(text, num_wrapping_dims=1)
        embeddings = self.backbone.embedder(text, num_wrapping_dims=1)
        logits = self._encoder_and_head_forward(embeddings, mask)

        output = self._make_forward_output(logits, label)

        # For computing the attributions
        # TODO: An optimized implementation would be to calculate the attributions directly in the forward method
        #  and provide a practical switch, maybe: `with head.turn_attributions_on(): self.forward_on_instances()`
        #  In this way we would calculate the attributions batch wise and on on GPU if available.
        output["embeddings"], output["mask"] = embeddings, mask

        return output

    def _encoder_and_head_forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """We reuse this method for the computation of the attributions"""
        encoded_text = self.tokens_pooler(
            self.backbone.encoder(embeddings, mask=mask), mask=mask
        )

        # Here we need to mask the TextFields that only contain the padding token -> last dimension only contains False
        # Those fields were added to possibly equalize the batch.
        mask = torch.sum(mask, -1) > 0
        encoded_text = self.sentences_pooler(
            self.sentences_encoder(encoded_text, mask=mask), mask=mask
        )

        if self.feedforward is not None:
            encoded_text = self.feedforward(encoded_text)

        return self._classification_layer(encoded_text)

    def _compute_attributions(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
        n_steps: int = 50,
    ) -> List[List[Attribution]]:
        """Attributes the prediction to the input.

        The attributions are calculated by means of the [Integrated Gradients](https://arxiv.org/abs/1703.01365) method.

        Parameters
        ----------
        single_forward_output
            Non-batched forward output containing numpy arrays
        instance
            The instance containing the input data
        n_steps
            The number of steps used when calculating the attribution of each token.

        Returns
        -------
        attributions
            A list of list of attributions due to the the ListField level
        """
        # captum needs `torch.Tensor`s and we need a batch dimension (-> unsqueeze)
        embeddings = torch.from_numpy(single_forward_output["embeddings"]).unsqueeze(0)
        mask = torch.from_numpy(single_forward_output["mask"]).unsqueeze(0)
        logits = torch.from_numpy(single_forward_output["logits"]).unsqueeze(0)

        ig = IntegratedGradients(self._encoder_and_head_forward)
        attributions, delta = ig.attribute(
            embeddings,
            n_steps=n_steps,
            target=torch.argmax(logits),
            additional_forward_args=mask,
            return_convergence_delta=True,
        )
        attributions = attributions.sum(dim=3).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().numpy()

        document_tokens = [
            cast(TextField, text_field).tokens
            for text_field in cast(ListField, instance.get(self.forward_arg_name))
        ]

        return [
            [
                Attribution(
                    text=token.text,
                    start=token.idx,
                    end=self._get_token_end(token),
                    field=self.forward_arg_name,
                    attribution=attribution,
                )
                for token, attribution in zip(sentence_tokens, sentence_attributions)
            ]
            for sentence_tokens, sentence_attributions in zip(
                document_tokens, attributions
            )
        ]

    def _make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
    ) -> DocumentClassificationPrediction:
        labels, probabilities = self._compute_labels_and_probabilities(
            single_forward_output
        )

        return DocumentClassificationPrediction(
            labels=labels, probabilities=probabilities
        )


class DocumentClassificationConfiguration(
    ComponentConfiguration[DocumentClassification]
):
    """Lazy initialization for document classification head components"""

    pass
