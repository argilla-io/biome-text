# fmt: off
from .defs import TaskHead, TaskHeadSpec, TaskName, TaskOutput
from .doc_classification import DocumentClassification
from .language_modelling import LanguageModelling, LanguageModellingSpec
from .record_classification import RecordClassification, RecordClassificationSpec
from .text_classification import TextClassification, TextClassificationSpec
from .token_classification import TokenClassification, TokenClassificationSpec
from .record_bimpm_classification import RecordBiMpm

# fmt: on

for head in [
    TextClassification,
    TokenClassification,
    DocumentClassification,
    RecordClassification,
    LanguageModelling,
    RecordBiMpm,
]:
    head.register(overrides=True)
