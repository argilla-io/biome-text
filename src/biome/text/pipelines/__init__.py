from ._impl.allennlp.classifier import TextClassifierPipeline
from .biome_bimpm import BiomeBiMpmPipeline
from .multifield_bimpm import MultifieldBiMpmPipeline
from .pipeline import Pipeline
from .sequence_classifier import SequenceClassifierPipeline
from .sequence_pair_classifier import SequencePairClassifierPipeline
from .similarity_classifier import SimilarityClassifierPipeline

for pipeline in [
    TextClassifierPipeline,
    BiomeBiMpmPipeline,
    MultifieldBiMpmPipeline,
    SequenceClassifierPipeline,
    SequencePairClassifierPipeline,
    SimilarityClassifierPipeline,
]:
    pipeline.init_class()
