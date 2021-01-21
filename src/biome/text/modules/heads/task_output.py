import dataclasses
from typing import Dict
from typing import List
from typing import Tuple
from typing import cast

# this sentinel is used to omit certain dataclass fields when the dataclass is converted to a dict
SENTINEL = cast(None, "SENTINEL TO SKIP DATACLASS FIELDS WHEN CONVERTING TO DICT")


@dataclasses.dataclass
class AttributionsOutput:
    text: str
    start: int
    end: int
    attribution: float


@dataclasses.dataclass
class TokensOutput:
    text: str
    start: int
    end: int


@dataclasses.dataclass
class TaskOutput:
    """Base class for the TaskOutput classes.

    Each head should implement a proper task output class that defines its prediction output.
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
class ClassificationOutput(TaskOutput):
    """Output dataclass for all `ClassificationHead`s:
    - `TextClassification`
    - `RecordClassification`
    - `DocumentClassification`
    - `RecordPairClassification`
    """

    labels: List[str]
    probabilities: List[float]
    attributions: AttributionsOutput = SENTINEL
    tokens: TokensOutput = SENTINEL


@dataclasses.dataclass
class TokenClassificationOutput(TaskOutput):
    """Output dataclass for the `TokenClassification` head"""
