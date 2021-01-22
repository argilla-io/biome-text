import dataclasses
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

import numpy

# this sentinel is used to omit certain dataclass fields when the dataclass is converted to a dict
SENTINEL = cast(None, "SENTINEL TO SKIP DATACLASS FIELDS WHEN CONVERTING TO DICT")


@dataclasses.dataclass
class Token:
    """Output dataclass for a token in a prediction.

    Parameters
    ----------
    text
        Text of the token
    start
        Start char id
    end
        End char id
    """

    text: str
    start: int
    end: int


@dataclasses.dataclass
class Entity:
    """Output dataclass for a NER entity

    Parameters
    ----------
    label
        Label of the entity
    start_token
        Start token id
    end_token
        End token id
    start
        Start char id
    end
        End char id
    """

    label: str
    start_token: int
    end_token: int
    start: Optional[int] = SENTINEL
    end: Optional[int] = SENTINEL


@dataclasses.dataclass
class TaskPrediction:
    """Base class for the TaskOutput classes.

    Each head should implement a proper task prediction class that defines its prediction output.
    You can use the SENTINEL as default value if you want to omit certain fields when converting to a dict.
    """

    @staticmethod
    def _dict_factory(key_value_pairs: List[Tuple]) -> Dict:
        """input is a list of (key, value) pairs"""
        return dict(
            [(key, value) for key, value in key_value_pairs if value is not SENTINEL]
        )

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self, dict_factory=self._dict_factory)


@dataclasses.dataclass
class ClassificationPrediction(TaskPrediction):
    """Output dataclass for all `ClassificationHead`s:
    - `TextClassification`
    - `RecordClassification`
    - `DocumentClassification`
    - `RecordPairClassification`

    Parameters
    ----------
    labels
        Ordered list of predictions, from the label with the highest to the label with the lowest probability.
    probabilities
        Ordered list of probabilities, from highest to lowest probability.
    """

    labels: List[str]
    probabilities: List[float]


@dataclasses.dataclass
class TokenClassificationPrediction(TaskPrediction):
    """Output dataclass for the `TokenClassification` head

    Parameters
    ----------
    tags
        List of lists of NER tags, ordered by score.
        The list of NER tags with the highest score comes first.
    entities
        List of list of entities, ordered by score.
        The list of entities with the highest score comes first.
    scores
        Ordered list of scores for each list of NER tags (highest to lowest).
    tokens
        Tokens of the tokenized input
    """

    tags: List[List[str]]
    entities: List[List[Entity]]
    scores: List[float]
    tokens: Optional[List[Token]] = SENTINEL


@dataclasses.dataclass
class LanguageModellingPrediction(TaskPrediction):
    """Output dataclass for the `LanguageModelling` head"""

    lm_embeddings: numpy.array
    mask: numpy.array
    # Is only included if batch size == 1
    loss: Optional[float] = SENTINEL
