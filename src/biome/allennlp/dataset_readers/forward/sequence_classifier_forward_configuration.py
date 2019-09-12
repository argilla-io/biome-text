from typing import Union, List

from biome.allennlp.dataset_readers.forward.classification_forward_configuration import (
    ClassificationForwardConfiguration,
)


class SequenceClassifierForwardConfiguration(ClassificationForwardConfiguration):
    """
        This ``SequenceClassifierForwardConfiguration`` extends basic forward
        configuration for ``SequenceClassifier``, by adding the `tokens` field configuration
    """

    def __init__(
        self,
        tokens: Union[str, List[str]],
        label: Union[str, dict] = None,
        target: dict = None,
    ):
        super(SequenceClassifierForwardConfiguration, self).__init__(label, target)
        self._tokens = [tokens] if isinstance(tokens, str) else tokens

    @property
    def tokens(self) -> List[str]:
        return self._tokens.copy()
