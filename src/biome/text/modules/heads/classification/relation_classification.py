import logging
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
from allennlp.data.fields import SequenceLabelField
from allennlp.data.fields import TextField
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask

from biome.text.backbone import ModelBackbone
from biome.text.featurizer import FeaturizeError
from biome.text.helpers import tags_from_offsets
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import EmbeddingConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.heads.classification.classification import ClassificationHead

# from biome.text.modules.encoders.multi_head_self_attention_encoder import MultiheadSelfAttentionEncoder
from biome.text.modules.heads.task_prediction import RelationClassificationPrediction


class RelationClassification(ClassificationHead):
    """
    Task head for relation classification

    Parameters
    ----------
    backbone
        The backbone of your model. Must not be provided when initiating with `Pipeline.from_config`.
    labels
        A list of labels for your classification task.
    entities_embedder
        The embedder for the entity tags.
    entity_encoding
        The encoding scheme of the entity tags. Default: BIOUL.
    pooler
        The pooler of the output sequence from the backbone model. Default: `BagOfEmbeddingsEncoder`.
    feedforward
        An optional feedforward layer applied to the output of the pooler. Default: None.
    multilabel
        Is this a multi label classification task? Default: False
    label_weights
        A list of weights for each label. The weights must be in the same order as the `labels`.
        You can also provide a dictionary that maps the label to its weight. Default: None.
    """

    _TEXT_ARG_NAME_IN_FORWARD = "text"
    _LABEL_ARG_NAME_IN_FORWARD = "label"
    _LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        entities_embedder: EmbeddingConfiguration,
        entity_encoding: Optional[str] = "BIOUL",
        pooler: Optional[Seq2VecEncoderConfiguration] = None,
        feedforward: Optional[FeedForwardConfiguration] = None,
        multilabel: bool = False,
        label_weights: Optional[Union[List[float], Dict[str, float]]] = None,
        # self_attention: Optional[MultiheadSelfAttentionEncoder] = None
    ) -> None:

        super().__init__(
            backbone=backbone,
            labels=labels,
            multilabel=multilabel,
            label_weights=label_weights,
        )

        self._label_encoding = entity_encoding
        self._entity_tags_namespace = "entities"

        self.entities_embedder = entities_embedder.compile()

        encoding_output_dim = (
            self.backbone.encoder.get_output_dim()
            + self.entities_embedder.get_output_dim()
        )
        self.pooler = (
            pooler.input_dim(encoding_output_dim).compile()
            if pooler
            else BagOfEmbeddingsEncoder(encoding_output_dim)
        )

        self.feedforward = (
            None
            if not feedforward
            else feedforward.input_dim(self.pooler.get_output_dim()).compile()
        )

        self._classification_layer = torch.nn.Linear(
            (self.feedforward or self.pooler).get_output_dim(), self.num_labels
        )
        # self.self_attention = self_attention

    def featurize(
        self,
        text: Union[str, List[str], Dict[str, str]],
        entities: List[Dict],
        label: Optional[Union[str, List[str]]] = None,
    ) -> Instance:
        instance = self.backbone.featurizer(
            text,
            to_field=self._TEXT_ARG_NAME_IN_FORWARD,
            aggregate=True,
            exclude_record_keys=True,
        )

        doc = self.backbone.tokenizer.nlp(text)
        entity_tags = tags_from_offsets(doc, entities, self._label_encoding)

        if "-" in entity_tags:
            raise FeaturizeError(
                f"Could not align spans with tokens for following example: '{text}' {entities}"
            )

        try:
            instance.add_field(
                "entities",
                SequenceLabelField(
                    entity_tags,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace=self._entity_tags_namespace,
                ),
            )
        except Exception as error:
            raise FeaturizeError(
                f"Could not create SequenceLabelField for {(text, entity_tags)}"
            ) from error

        return self._add_label(
            instance, label, to_field=self._LABEL_ARG_NAME_IN_FORWARD
        )

    def forward(  # type: ignore
        self,
        text: TextFieldTensors,
        entities: torch.IntTensor,
        label: torch.IntTensor = None,
    ) -> Dict[str, Any]:

        mask = get_text_field_mask(text)
        embedded_text = self.backbone.forward(text, mask)

        embedded_ents = self.entities_embedder(entities)
        embedded_text = torch.cat((embedded_text, embedded_ents), dim=-1)
        """
        if self.self_attention:
            embedded_text = self.self_attention(embedded_text)
        """
        embedded_text = self.pooler(embedded_text, mask=mask)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)

        return self._make_forward_output(logits=logits, label=label)

    def _make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
    ) -> RelationClassificationPrediction:
        labels, probabilities = self._compute_labels_and_probabilities(
            single_forward_output
        )

        return RelationClassificationPrediction(
            labels=labels, probabilities=probabilities
        )


class RelationClassificationConfiguration(
    ComponentConfiguration[RelationClassification]
):
    """Configuration for classification head components"""

    pass
