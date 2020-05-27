from .task_head import TaskHead, TaskHeadSpec, TaskName, TaskOutput

from .classification.doc_classification import DocumentClassification
from .classification.record_classification import RecordClassification
from .classification.record_pair_classification import RecordPairClassification
from .classification.text_classification import TextClassification

from .language_modelling import LanguageModelling, LanguageModellingSpec
from .token_classification import TokenClassification, TokenClassificationSpec


for head in [
    TextClassification,
    TokenClassification,
    DocumentClassification,
    RecordClassification,
    LanguageModelling,
    RecordPairClassification,
]:
    head.register(overrides=True)
