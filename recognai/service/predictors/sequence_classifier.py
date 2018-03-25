import csv

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors import Predictor
from overrides import overrides
from typing import Tuple

from recognai.data.dataset_readers import ClassificationDatasetReader


@Predictor.register('sequence-classifier')
class SequenceClassifierPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader):
        assert (dataset_reader.__class__ == ClassificationDatasetReader,
                "Wrong configuration for prediction. This kind of predictor only accept an classification dataset reader")

        super().__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        instance = self._dataset_reader.process_example(json_dict)
        return instance, {"input": json_dict}
