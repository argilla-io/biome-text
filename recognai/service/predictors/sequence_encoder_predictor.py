import csv

from typing import List, Tuple, Any

import torch, numpy, spacy, allennlp

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.data.dataset import Batch
from allennlp.service.predictors import Predictor
from overrides import overrides
from typing import Tuple

from recognai.data.dataset_readers import ClassificationDatasetReader


@Predictor.register('sequence_encoder')
class SequenceEncoderPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        assert isinstance(dataset_reader, ClassificationDatasetReader), \
            "Wrong configuration for prediction." \
            + "This kind of predictor only accept an classification dataset reader"

        super(SequenceEncoderPredictor, self).__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        instance = self._dataset_reader.process_example(json_dict)
        return instance, {"input": json_dict}
    
    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device=None) -> List[JsonDict]:
        instances, return_dicts = zip(*self._batch_json_to_instances(inputs))
        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        model_input = dataset.as_tensor_dict()
        encoded_records = self._model.encode_tokens(model_input['record_1'])
        
        for output, return_dict in zip(encoded_records, return_dicts):
            return_dict.update({ "vector": output.data })
        return sanitize(return_dicts)

# TODO: remove when updating allen to 0.45
def sanitize(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif isinstance(x, (spacy.tokens.Token, allennlp.data.Token)):
        # Tokens get sanitized to just their text.
        return x.text
    elif x is None:
        return "None"
    elif hasattr(x, 'to_json'):
        return x.to_json()
    else:
        raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                        "If this is your own custom class, add a `to_json(self)` method "
        "that returns a JSON-like object.")