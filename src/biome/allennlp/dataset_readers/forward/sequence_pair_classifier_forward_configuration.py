from typing import List, Union

from biome.allennlp.dataset_readers.forward.classification_forward_configuration import (
    ClassificationForwardConfiguration,
)


class SequencePairClassifierForwardConfiguration(ClassificationForwardConfiguration):
    """
        This ``SequencePairClassifierForwardConfiguration`` extends forward
        configuration for ``SequencePairClassifier``, by adding the `record1` and `record2` fields configuration
    """

    def __init__(
        self,
        record1: Union[str, List[str]],
        record2: Union[str, List[str]],
        label: Union[str, dict] = None,
        target: dict = None,
    ):
        super(SequencePairClassifierForwardConfiguration, self).__init__(label, target)
        self._record1 = [record1] if isinstance(record1, str) else record1
        self._record2 = [record2] if isinstance(record2, str) else record2

    @property
    def record1(self) -> List[str]:
        return self._record1

    @property
    def record2(self) -> List[str]:
        return self._record2
