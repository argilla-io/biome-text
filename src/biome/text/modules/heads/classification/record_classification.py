from typing import List
from typing import Optional
from typing import Union

from allennlp.data import Instance

from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2SeqEncoderConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration

from .doc_classification import DocumentClassification


class RecordClassification(DocumentClassification):
    """
    Task head for data record  classification.
    Accepts a variable data inputs and apply featuring over defined record keys.

    This head applies a doc2vec architecture from a structured record data input
    """

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        record_keys: List[str],
        tokens_pooler: Optional[Seq2VecEncoderConfiguration] = None,
        fields_encoder: Optional[Seq2SeqEncoderConfiguration] = None,
        fields_pooler: Optional[Seq2VecEncoderConfiguration] = None,
        feedforward: Optional[FeedForwardConfiguration] = None,
        multilabel: Optional[bool] = False,
    ) -> None:

        super(RecordClassification, self).__init__(
            backbone,
            labels=labels,
            tokens_pooler=tokens_pooler,
            sentences_encoder=fields_encoder,
            sentences_pooler=fields_pooler,
            feedforward=feedforward,
            multilabel=multilabel,
        )

        self._inputs = record_keys

    def inputs(self) -> Optional[List[str]]:
        """The inputs names are determined by configured record keys"""
        return self._inputs

    def featurize(
        self, label: Optional[Union[List[str], List[int], str, int]] = None, **inputs
    ) -> Optional[Instance]:

        instance = self.backbone.featurizer(
            {input_key: inputs[input_key] for input_key in self._inputs},
            to_field="document",
        )
        return self.add_label(instance, label)


class RecordClassificationConfiguration(ComponentConfiguration[RecordClassification]):
    """Lazy initialization for document classification head components"""

    pass
