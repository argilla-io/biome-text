import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

import torch
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.data.fields import SequenceLabelField
from allennlp.data.fields import TextField
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask

from biome.text.backbone import ModelBackbone
from biome.text.helpers import tags_from_offsets
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import EmbeddingConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration

from ..task_head import TaskOutput
from .text_classification import TextClassification

# from biome.text.modules.encoders.multi_head_self_attention_encoder import MultiheadSelfAttentionEncoder


class RelationClassification(TextClassification):
    """
    Task head for relation classification
    """

    __LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        entities_embedder: EmbeddingConfiguration,
        pooler: Optional[Seq2VecEncoderConfiguration] = None,
        feedforward: Optional[FeedForwardConfiguration] = None,
        multilabel: bool = False,
        entity_encoding: Optional[str] = "BIOUL"
        # self_attention: Optional[MultiheadSelfAttentionEncoder] = None
    ) -> None:

        super(RelationClassification, self).__init__(
            backbone,
            labels,
            pooler=pooler,
            feedforward=feedforward,
            multilabel=multilabel,
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

    def forward(  # type: ignore
        self,
        text: TextFieldTensors,
        entities: torch.IntTensor,
        label: torch.IntTensor = None,
    ) -> TaskOutput:

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
        return self.calculate_output(logits=logits, label=label)

    def featurize(
        self,
        text: Any,
        entities: List[Dict],
        label: Optional[Union[int, str, List[Union[int, str]]]] = None,
    ) -> Optional[Instance]:

        instance = self.backbone.featurizer(
            text,
            to_field=self.forward_arg_name,
            aggregate=True,
            exclude_record_keys=True,
        )

        doc = self.backbone.tokenizer.nlp(text)
        entity_tags = tags_from_offsets(doc, entities, self._label_encoding)

        if "-" in entity_tags:
            self.__LOGGER.warning(
                f"Could not align spans with tokens for following example: '{text}' {entities}"
            )
            return None

        instance.add_field(
            "entities",
            SequenceLabelField(
                entity_tags,
                sequence_field=cast(TextField, instance["text"]),
                label_namespace=self._entity_tags_namespace,
            ),
        )

        return self.add_label(instance, label, to_field=self.label_name)


class RelationClassificationConfiguration(
    ComponentConfiguration[RelationClassification]
):
    """Configuration for classification head components"""

    pass
