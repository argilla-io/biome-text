from typing import List
from typing import Optional

from allennlp.data import Instance
from transformers import AutoTokenizer

from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.heads import TaskHead


class ProfNer(TaskHead):
    def __init__(
        self,
        backbone: ModelBackbone,
        classification_labels: List[str],
        ner_tags: List[str],
        transformers_model: Optional[str] = None,
        label_encoding: Optional[str] = "BIOUL",
        top_k: int = 1,
        dropout: Optional[float] = 0.0,
        feedforward: Optional[FeedForwardConfiguration] = None,
    ) -> None:
        super().__init__(backbone)

        self._transformers_model = transformers_model

    def featurize(self, text: List[str], labels: str, tags: List[str]) -> Instance:
        """

        Parameters
        ----------
        text
            pretokenized input
        labels
            label
        tags
            BIO tags

        Returns
        -------
        instance

        Raises
        ------
        FeaturizeError
        """
        if self._transformers_model:
            tokenizer = AutoTokenizer.from_pretrained(self._transformers_model)
            input_ids = tokenizer(
                text, is_split_into_words=True, return_offsets_mapping=True
            )


class ProfNerConfiguration(ComponentConfiguration[ProfNer]):
    """Configuration for classification head components"""

    pass
