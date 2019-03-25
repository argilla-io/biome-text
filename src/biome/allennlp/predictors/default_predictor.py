from typing import List

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


class DefaultBasePredictor(Predictor):

    def __init__(self, model: Model, reader: DatasetReader) -> None:
        super(DefaultBasePredictor, self).__init__(model, reader)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output = self.predict_instance(instance)
        return self.__to_prediction(inputs, output)

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instance = self._json_to_instance(json_dict)
            # skip examples that failed to be converted to instances! For example (and maybe solely) empty strings
            if instance:
                instances.append(instance)
        return instances

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self._model.forward_on_instances(instances)
        return [self.__to_prediction(input, output) for input, output in zip(inputs, outputs)]

    @staticmethod
    def __to_prediction(inputs, output):
        return dict(input=inputs, annotation=sanitize(output))


@Predictor.register("sequence_classifier")
class SequenceClassifierPredictor(DefaultBasePredictor):
    pass
