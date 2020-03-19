from typing import Union, List

from allennlp.predictors import Predictor

from biome.text.dataset_readers import SequencePairClassifierReader
from .pipeline import Pipeline
from ..models import SequencePairClassifier


class SequencePairClassifierPipeline(
    Pipeline[SequencePairClassifier, SequencePairClassifierReader]
):

    # pylint: disable=arguments-differ
    def predict(
        self, record1: Union[str, List[str], dict], record2: Union[str, List[str], dict]
    ):
        instance = self.reader.text_to_instance(record1=record1, record2=record2)
        return self.model.forward_on_instance(instance)


Predictor.register("sequence_pair_classifier")(SequencePairClassifierPipeline)
Predictor.register(SequencePairClassifierPipeline.__name__)(
    SequencePairClassifierPipeline
)
