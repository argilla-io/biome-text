# fmt: off
from .defs import TaskHead, TaskName, TaskOutput
from .doc_classification import DocumentClassification
from .record_classification import RecordClassification, RecordClassificationSpec
from .text_classification import TextClassification, TextClassificationSpec
from .token_classification import TokenClassification, TokenClassificationSpec
from .language_modelling import LanguageModelling, LanguageModellingSpec
from .bimpm_classification import BiMpm
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
