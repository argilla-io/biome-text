from copy import deepcopy
from typing import List, Dict

import numpy

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.data.fields import LabelField
#from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient
from overrides import overrides


class DefaultBasePredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)

        # check if instance is valid and print out meaningful error message
        if instance is None:
            raise ValueError(
                "Input {} could not be converted to an instance. No prediction possible.".format(
                    inputs
                )
            )

        output = self.predict_instance(instance)
        return self._to_prediction(inputs, output)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        
        # I tried to work with numpy arrays here to elegantly mask the Nones ... does not work with instances!!!
        # This fails: np.array(instances)

        # skip examples that failed to be converted to instances! For example (and maybe solely) empty strings
        # !this results in an empty list if `inputs` is a generator!
        input_and_instances = [
            (input, instance)
            for input, instance in zip(inputs, instances)
            if instance is not None
        ]

        # check if there are instances left and print out meaningful error message
        if not input_and_instances:
            raise IndexError(
                "No instances found in batch. Check input or make batch size bigger."
            )

        outputs = self._model.forward_on_instances(
            [instance[1] for instance in input_and_instances]
        )

        return [
            self._to_prediction(input[0], output)
            for input, output in zip(input_and_instances, outputs)
        ]

    @staticmethod
    def _to_prediction(input, output):
        output.pop("logits")
        return {**input, "annotation": sanitize(output)}
    
    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        
        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs["logits"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))

        return [new_instance]


@Predictor.register("sequence_classifier")
class SequenceClassifierPredictor(DefaultBasePredictor):
    pass
