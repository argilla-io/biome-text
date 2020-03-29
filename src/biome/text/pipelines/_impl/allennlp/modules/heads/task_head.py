from typing import Dict

import torch
from allennlp.data import Vocabulary

from biome.text.pipelines._impl.allennlp.models.defs import WithLayerChain


class TaskHead(torch.nn.Module, WithLayerChain):
    """Base task head for last `TextClassifier` model layer"""

    def __init__(self, vocab=Vocabulary):
        super(TaskHead, self).__init__()
        self.vocab = vocab

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        raise NotImplementedError
