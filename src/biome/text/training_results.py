from dataclasses import dataclass
from typing import Any
from typing import Dict


@dataclass()
class TrainingResults:
    """
    Training results data class

    Attributes
    ----------

    model_path: `str`
        The trained model path
    metrics: `Dict[str, Any]`
        Related training metrics

    """

    model_path: str
    metrics: Dict[str, Any]
