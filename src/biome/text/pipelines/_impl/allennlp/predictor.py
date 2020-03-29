import warnings
from copy import deepcopy
from typing import List, Optional, cast, Tuple, Dict, Any

import numpy as np
from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.data.fields import LabelField
from allennlp.models import Model
from allennlp.predictors import Predictor

from biome.text.pipelines._impl.allennlp.models.defs import ITextClassifier


class TextClassifierPredictor(Predictor):
    """Text classifier predictor. This predictor will support for ITextClassifier model implementations"""

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        if not isinstance(model, ITextClassifier):
            raise TypeError(
                f"Cannot support model of type {model.__class__}. Only for ITextClassifier implementations"
            )

        super(TextClassifierPredictor, self).__init__(model, dataset_reader)

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        """
        Converts incoming json to a :class:`~allennlp.data.instance.Instance`,
        runs the model on the newly created instance, and adds labels to the
        :class:`~allennlp.data.instance.Instance`s given by the model's output.
        Returns
        -------
        List[instance]
        A list of :class:`~allennlp.data.instance.Instance`
        """
        # pylint: disable=assignment-from-no-return
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    def predict_json(self, inputs: JsonDict) -> Optional[JsonDict]:
        """Predict an input with the pipeline's model.

        Parameters
        ----------
        inputs
            The input features/tokens in form of a json dict

        Returns
        -------
        output
            The model's prediction in form of a dict.
            Returns None if the input could not be transformed to an instance.
        """
        instance = self._json_to_instance(inputs)
        if instance is None:
            return None
        output = sanitize(self._model.forward_on_instance(instance))
        return output

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, np.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        # TODO: Open question to @dvilasuero
        #  Keep the following code related to the `allennlp.model.Model` head layer ?
        #  In this case, we need keep in mind the interpretation details in pipeline design
        label = np.argmax(outputs["logits"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))

        return [new_instance]

    def get_gradients(
        self, instances: List[Instance]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        Parameters
        ----------
        instances: List[Instance]

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
        The first item is a Dict of gradient entries for each input.
        The keys have the form  ``{grad_input_1: ..., grad_input_2: ... }``
        up to the number of inputs given. The second item is the model's output.

        Notes
        -----
        Takes a ``JsonDict`` representing the inputs of the model and converts
        them to :class:`~allennlp.data.instance.Instance`s, sends these through
        the model :func:`forward` function after registering hooks on the embedding
        layer of the model. Calls :func:`backward` on the loss and then removes the
        hooks.
        """
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        outputs = self._model.decode(self._model.forward(**dataset.as_tensor_dict()))

        loss = outputs["loss"]
        self._model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        embedding_gradients.reverse()
        grads = [grad.detach().cpu().numpy() for grad in embedding_gradients]
        return grads, outputs

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(**json_dict)

    def extend_labels(self, labels: List[str]) -> None:
        """Allow extend prediction labels to pipeline"""
        if not isinstance(self._model, ITextClassifier):
            warnings.warn(f"Model {self._model} is not updatable")
        else:
            cast(ITextClassifier, self._model).extend_labels(labels)

    def get_output_labels(self) -> List[str]:
        """Return the prediction classes"""
        if not isinstance(self._model, ITextClassifier):
            warnings.warn(f"get_output_labels not supported for model {self._model}")
            return []
        return cast(ITextClassifier, self._model).output_classes()
