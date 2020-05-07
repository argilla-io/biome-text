# fmt: off
from .bimpm_classification import BiMpm
from .defs import TaskHead, TaskHeadSpec, TaskName, TaskOutput
from .doc_classification import DocumentClassification
from .language_modelling import LanguageModelling, LanguageModellingSpec
from .record_classification import RecordClassification, RecordClassificationSpec
from .text_classification import TextClassification, TextClassificationSpec
from .token_classification import TokenClassification, TokenClassificationSpec

# fmt: on

for head in [
    TextClassification,
    TokenClassification,
    DocumentClassification,
    RecordClassification,
    LanguageModelling,
    BiMpm,
]:
    head.register(overrides=True)
