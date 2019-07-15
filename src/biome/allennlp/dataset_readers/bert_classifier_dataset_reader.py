import logging
from inspect import signature
from typing import Dict

from allennlp.data import DatasetReader, TokenIndexer, Tokenizer

from biome.allennlp.dataset_readers import SequenceClassifierDatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("bert_for_classification")
class BertClassifierDatasetReader(SequenceClassifierDatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        from allennlp.models import BertForClassification

        super(BertClassifierDatasetReader, self).__init__(
            tokenizer=tokenizer, token_indexers=token_indexers
        )

        # The keys of the Instances have to match the signature of the forward method of the model
        self.forward_params = signature(BertForClassification.forward).parameters
