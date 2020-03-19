from typing import Union, List

from allennlp.predictors import Predictor

from biome.text.dataset_readers import SequencePairClassifierReader
from biome.text.models import SimilarityClassifier as TorchSimilarityClassifier
from biome.text.pipelines import SequencePairClassifierPipeline
from biome.text.pipelines.pipeline import Pipeline


class SimilarityClassifierPipeline(
    Pipeline[TorchSimilarityClassifier, SequencePairClassifierReader]
):
    predict = SequencePairClassifierPipeline.predict


Predictor.register("similarity_classifier")(SimilarityClassifierPipeline)
Predictor.register(SimilarityClassifierPipeline.__name__)(SimilarityClassifierPipeline)
