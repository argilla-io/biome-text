from typing import Union, List, Type

from allennlp.models import Model as AllennlpModel
from allennlp.predictors import Predictor

from biome.text.pipelines.pipeline import Pipeline
from biome.text.dataset_readers import SequencePairClassifierReader
from biome.text.dataset_readers.datasource_reader import DataSourceReader
from biome.text.models import SimilarityClassifier as TorchSimilarityClassifier


class SimilarityClassifier(Pipeline):
    @classmethod
    def reader_class(cls) -> Type[DataSourceReader]:
        return SequencePairClassifierReader

    @classmethod
    def model_class(cls) -> Type[AllennlpModel]:
        return TorchSimilarityClassifier

    # pylint: disable=arguments-differ
    def predict(
        self, record1: Union[str, List[str], dict], record2: Union[str, List[str], dict]
    ):
        instance = self.reader.text_to_instance(record1=record1, record2=record2)
        return self.model.forward_on_instance(instance)


Predictor.register("similarity_classifier")(SimilarityClassifier)
Predictor.register(SimilarityClassifier.__name__)(SimilarityClassifier)
