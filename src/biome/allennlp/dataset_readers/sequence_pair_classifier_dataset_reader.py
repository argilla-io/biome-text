import logging
from inspect import signature
from typing import Dict

from allennlp.data import DatasetReader, TokenIndexer, Tokenizer

from biome.allennlp.models import SequencePairClassifier
from .sequence_classifier_dataset_reader import SequenceClassifierDatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sequence_pair_classifier")
class SequencePairClassifierDatasetReader(SequenceClassifierDatasetReader):
    """A DatasetReader for the SequencePairClassifier model.

    Parameters
    ----------
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter
    token_indexers
        By default we use a SingleIdTokenIndexer for all token fields
    as_text_field
        False by default, if enabled, the ``Instances`` generated will contains
        the input text fields as a concatenation of input fields in a single ``TextField``.
        By default, the reader will use a ``ListField`` of ``TextField`` for input representation
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        as_text_field: bool = False,
    ) -> None:
        super(SequencePairClassifierDatasetReader, self).__init__(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            as_text_field=as_text_field,
        )

        # The keys of the Instances have to match the signature of the forward method of the model
        self.forward_params = self.get_forward_signature(SequencePairClassifier)
