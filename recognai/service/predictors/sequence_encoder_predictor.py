import csv

from typing import List, Tuple

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
        record_1 = model_input['record_1']
        encoded_record_1 = self._model.encode_tokens(record_1)
        return encoded_record_1