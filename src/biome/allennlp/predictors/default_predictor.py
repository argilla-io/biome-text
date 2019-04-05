from typing import List

import numpy as np
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

        # check if instance is valid and print out meaningful error message
        if instance is None:
            raise ValueError("Input {} could not be converted to an instance. No prediction possible.".format(inputs))

        output = self.predict_instance(instance)
        return self._to_prediction(inputs, output)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = np.array(self._batch_json_to_instances(inputs))
        inputs_ar = np.array(inputs)

        # skip examples that failed to be converted to instances! For example (and maybe solely) empty strings
        idx = instances != np.array(None)

        # check if there are instances left and print out meaningful error message
        if not any(idx):
            raise IndexError("No instances found in batch. Check input or make batch size bigger.")

        outputs = self._model.forward_on_instances(instances[idx])
        return [
            self._to_prediction(input, output)
            for input, output in zip(inputs_ar[idx], outputs)
        ]

    @staticmethod
    def _to_prediction(inputs, output):
        return dict(input=inputs, annotation=sanitize(output))


@Predictor.register("sequence_classifier")
class SequenceClassifierPredictor(DefaultBasePredictor):
    pass
