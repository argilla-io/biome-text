from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor

from biome.text.pipelines._impl.allennlp.dataset_readers import (
    SequencePairClassifierReader,
)
from biome.text.pipelines._impl.allennlp.models.biome_bimpm import BiomeBiMpm
from biome.text.pipelines.pipeline import Pipeline
from .sequence_pair_classifier import SequencePairClassifierPipeline


class BiomeBiMpmPipeline(Pipeline[BiomeBiMpm, SequencePairClassifierReader]):
    predict = SequencePairClassifierPipeline.predict


# TODO: We register everything here, since the idea is to use the Pipeline class for all our commands
#       -> you always have to define a pipeline class -> all components should be registered under the same name!
Predictor.register("biome_bimpm")(BiomeBiMpmPipeline)
Model.register("biome_bimpm")(BiomeBiMpm)
DatasetReader.register("biome_bimpm")(SequencePairClassifierReader)
