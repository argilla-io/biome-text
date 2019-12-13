# pylint: disable=protected-access
import math
from typing import List, Dict, Any

import numpy

import operator

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.integrated_gradient import (
    IntegratedGradient,
)
from allennlp.interpret.saliency_interpreters.saliency_interpreter import (
    SaliencyInterpreter,
)
from allennlp.nn import util


@SaliencyInterpreter.register("biome-integrated-gradient")
class IntegratedGradient(IntegratedGradient):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)
    """

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        def normalize_gradient(grad):
            embedding_grad = numpy.sum(grad, axis=1)
            norm = numpy.linalg.norm(embedding_grad, ord=1)
            return [math.fabs(e) / norm for e in embedding_grad]

        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads: List = []
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self._integrate_gradients(instance)

            # Normalize results
            grads = [normalize_gradient(grad[0]) for grad in grads]
            # TODO: We might move this responsibility to the Predictor
            # or pass the order of fields when calling interpret.
            grads = self._pack_with_field_tokens(instance, grads)
            instances_with_grads.append(grads)

        return sanitize(instances_with_grads)

    def _pack_with_field_tokens(self, instance: Instance, grads):
        # We assume keys are ordered in the same way fields are processed
        list_keys = list(instance.fields.keys())
        list_keys.remove("label")
        output: Dict = {}
        for idx, key in enumerate(list_keys):
            output[key] = [
                {"token": token, "grad": grad}
                for token, grad in zip(instance[key].tokens, grads[idx])
            ]
        return output

    def _integrate_gradients(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Returns integrated gradients for the given :class:`~allennlp.data.instance.Instance`
        """
        ig_grads: List[Any] = []

        # List of Embedding inputs
        embeddings_list: List[numpy.ndarray] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads = self.predictor.get_gradients([instance])[0]
            handle.remove()

            # Running sum of gradients
            if ig_grads == []:
                ig_grads = grads
            else:
                for idx, grad in enumerate(grads):
                    ig_grads[idx] += grad

        # Average of each gradient term
        ig_grads = [v / steps for v in ig_grads]

        # Element-wise multiply average gradient by the input
        multiply = lambda a, b: map(operator.mul, a, b)

        return list(multiply(ig_grads, embeddings_list))
