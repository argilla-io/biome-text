from typing import Tuple

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors import Predictor
from overrides import overrides
from recognai.data.dataset_readers import ClassificationDatasetReader


@Predictor.register('sequence-classifier')
class SequenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        assert isinstance(dataset_reader, ClassificationDatasetReader), \
            "Wrong configuration for prediction." \
            + "This kind of predictor only accept an classification dataset reader"

        super(SequenceClassifierPredictor, self).__init__(model, dataset_reader)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance, _ = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)

        return {
            'input': inputs,
            'annotation': sanitize(outputs)
        }

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        instance = self._dataset_reader.process_example(json_dict)
        return instance, None
