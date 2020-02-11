from typing import Union, List, Type

import allennlp
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor

from biome.text.dataset_readers import SequencePairClassifierReader
from biome.text.dataset_readers.datasource_reader import DataSourceReader
from biome.text.models import MultifieldBiMpm
from biome.text.pipelines.pipeline import Pipeline


class MultifieldBiMpmPipeline(Pipeline):
    @classmethod
    def reader_class(cls) -> Type[DataSourceReader]:
        return SequencePairClassifierReader

    @classmethod
    def model_class(cls) -> Type[allennlp.models.Model]:
        return MultifieldBiMpm

    # pylint: disable=arguments-differ
    def predict(
        self, record1: Union[str, List[str], dict], record2: Union[str, List[str], dict]
    ):
        instance = self.reader.text_to_instance(record1=record1, record2=record2)
        return self.model.forward_on_instance(instance)


# TODO: We register everything here, since the idea is to use the Pipeline class for all our commands
#       -> you always have to define a pipeline class -> all components should be registered under the same name!
Predictor.register("multifield_bimpm")(MultifieldBiMpmPipeline)
Model.register("multifield_bimpm")(MultifieldBiMpm)
DatasetReader.register("multifield_bimpm")(SequencePairClassifierReader)
