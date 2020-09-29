from .task_head import TaskHead, TaskHeadConfiguration, TaskName, TaskOutput

from .classification.doc_classification import (
    DocumentClassification,
    DocumentClassificationConfiguration,
)
from .classification.record_classification import (
    RecordClassification,
    RecordClassificationConfiguration,
)
from .classification.record_pair_classification import (
    RecordPairClassification,
    RecordPairClassificationConfiguration,
)
from .classification.text_classification import (
    TextClassification,
    TextClassificationConfiguration,
)
from .classification.relation_classification import (
    RelationClassification,
    RelationClassificationConfiguration,
)

from .language_modelling import LanguageModelling, LanguageModellingConfiguration
from .token_classification import TokenClassification, TokenClassificationConfiguration


for head in [
    TextClassification,
    TokenClassification,
    DocumentClassification,
    RecordClassification,
    LanguageModelling,
    RecordPairClassification,
    RelationClassification
]:
    head.register(overrides=True)
