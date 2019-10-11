from typing import Union, List, Type

import allennlp

import biome
from biome.text import Pipeline
from biome.text.dataset_readers import SequencePairClassifierDatasetReader
from biome.text.dataset_readers.datasource_reader import DataSourceReader


class SequencePairClassifier(Pipeline):
    @classmethod
    def reader_class(cls) -> Type[DataSourceReader]:
        return SequencePairClassifierDatasetReader

    @classmethod
    def model_class(cls) -> Type[allennlp.models.Model]:
        return biome.text.models.SequencePairClassifier

    def predict(
        self, record1: Union[str, List[str], dict], record2: Union[str, List[str], dict]
    ):
        instance = self.pipeline.text_to_instance(record1=record1, record2=record2)
        return self.architecture.forward_on_instance(instance)
