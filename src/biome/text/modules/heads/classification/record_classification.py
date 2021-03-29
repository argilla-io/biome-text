from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy
from allennlp.data import Instance

from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2SeqEncoderConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.heads import DocumentClassification
from biome.text.modules.heads.task_prediction import RecordClassificationPrediction


class RecordClassification(DocumentClassification):
    """
    Task head for data record  classification.
    Accepts a variable data inputs and apply featuring over defined record keys.

    This head applies a doc2vec architecture from a structured record data input

    Parameters
    ----------
    backbone
        The backbone of your model. Must not be provided when initiating with `Pipeline.from_config`.
    labels
        A list of labels for your classification task.
    token_pooler
        The pooler at token level to provide one vector per record field. Default: `BagOfEmbeddingsEncoder`.
    fields_encoder
        An optional sequence to sequence encoder that contextualizes the record field representations. Default: None.
    fields_pooler
        The pooler at sentence level to provide a vector for the whole record. Default: `BagOfEmbeddingsEncoder`.
    feedforward
        An optional feedforward layer applied to the output of the fields pooler. Default: None.
    multilabel
        Is this a multi label classification task? Default: False
    label_weights
        A list of weights for each label. The weights must be in the same order as the `labels`.
        You can also provide a dictionary that maps the label to its weight. Default: None.
    """

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        record_keys: List[str],
        token_pooler: Optional[Seq2VecEncoderConfiguration] = None,
        fields_encoder: Optional[Seq2SeqEncoderConfiguration] = None,
        fields_pooler: Optional[Seq2VecEncoderConfiguration] = None,
        feedforward: Optional[FeedForwardConfiguration] = None,
        multilabel: Optional[bool] = False,
        label_weights: Optional[Union[List[float], Dict[str, float]]] = None,
    ) -> None:

        super().__init__(
            backbone,
            labels=labels,
            token_pooler=token_pooler,
            sentence_encoder=fields_encoder,
            sentence_pooler=fields_pooler,
            feedforward=feedforward,
            multilabel=multilabel,
            label_weights=label_weights,
        )

        self._inputs = record_keys

    def inputs(self) -> Optional[List[str]]:
        """The inputs names are determined by configured record keys"""
        return self._inputs

    def featurize(
        self, label: Optional[Union[str, List[str]]] = None, **inputs
    ) -> Instance:

        record = {input_key: inputs[input_key] for input_key in self._inputs}
        instance = self.backbone.featurizer(record, to_field=self.forward_arg_name)

        return self._add_label(instance, label)

    def _make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
    ) -> RecordClassificationPrediction:
        labels, probabilities = self._compute_labels_and_probabilities(
            single_forward_output
        )

        return RecordClassificationPrediction(
            labels=labels, probabilities=probabilities
        )


class RecordClassificationConfiguration(ComponentConfiguration[RecordClassification]):
    """Lazy initialization for document classification head components"""

    pass
