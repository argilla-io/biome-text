from typing import List, Optional, Union

from allennlp.data import Instance

from biome.text.model import Model
from biome.text.modules.specs import (
    ComponentSpec,
    FeedForwardSpec,
    Seq2SeqEncoderSpec,
    Seq2VecEncoderSpec,
)
from .doc_classification import DocumentClassification


class RecordClassification(DocumentClassification):
    """
    Task head for data record  classification.
    Accepts a variable data inputs and apply featuring over defined record keys.

    This head applies a doc2vec architecture from a structured record data input
    """

    def __init__(
        self,
        model: Model,
        pooler: Seq2VecEncoderSpec,
        labels: List[str],
        record_keys: List[str],
        tokens_pooler: Optional[Seq2VecEncoderSpec] = None,
        encoder: Optional[Seq2SeqEncoderSpec] = None,
        feedforward: Optional[FeedForwardSpec] = None,
        multilabel: bool = False,
    ) -> None:

        super(RecordClassification, self).__init__(
            model,
            pooler,
            labels=labels,
            tokens_pooler=tokens_pooler,
            encoder=encoder,
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

        instance = self.model.featurize(
            {input_key: inputs[input_key] for input_key in self._inputs},
            to_field="document",
        )
        return self.add_label(instance, label)


class RecordClassificationSpec(ComponentSpec[RecordClassification]):
    """Lazy initialization for document classification head components"""

    pass
