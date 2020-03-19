from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor

from biome.text.dataset_readers import SequencePairClassifierReader
from biome.text.models import MultifieldBiMpm
from biome.text.pipelines import SequencePairClassifierPipeline
from biome.text.pipelines.pipeline import Pipeline


class MultifieldBiMpmPipeline(Pipeline[MultifieldBiMpm, SequencePairClassifierReader]):
    predict = SequencePairClassifierPipeline.predict


# TODO: We register everything here, since the idea is to use the Pipeline class for all our commands
#       -> you always have to define a pipeline class -> all components should be registered under the same name!
Predictor.register("multifield_bimpm")(MultifieldBiMpmPipeline)
Model.register("multifield_bimpm")(MultifieldBiMpm)
DatasetReader.register("multifield_bimpm")(SequencePairClassifierReader)
