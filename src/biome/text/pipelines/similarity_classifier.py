from typing import Union, List, Type

import allennlp
from allennlp.predictors import Predictor

import biome
from biome.text.dataset_readers import SequencePairClassifierDatasetReader
from biome.text.dataset_readers.datasource_reader import DataSourceReader
from .pipeline import Pipeline


class SimilarityClassifier(Pipeline):
    @classmethod
    def reader_class(cls) -> Type[DataSourceReader]:
        return SequencePairClassifierDatasetReader

    @classmethod
    def model_class(cls) -> Type[allennlp.models.Model]:
        return biome.text.models.SimilarityClassifier

    def predict(
        self, record1: Union[str, List[str], dict], record2: Union[str, List[str], dict]
    ):
        instance = self.reader.text_to_instance(record1=record1, record2=record2)
        return self.model.forward_on_instance(instance)


Predictor.register("similarity_classifier")(SimilarityClassifier)
Predictor.register(SimilarityClassifier.__name__)(SimilarityClassifier)
