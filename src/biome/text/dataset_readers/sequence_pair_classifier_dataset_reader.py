import logging
from typing import Dict, Optional, Union, List

from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance
from allennlp.data.fields import LabelField

from biome.text.models import SequencePairClassifier
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

    def text_to_instance(
        self,
        record1: Union[str, List[str], dict],
        record2: Union[str, List[str], dict],
        label: Optional[str] = None,
        **metadata
    ) -> Optional[Instance]:

        fields = {}

        record1_field = self.build_textfield(record1)
        record2_field = self.build_textfield(record2)
        label_field = LabelField(label) if label else None

        if record1_field:
            fields["record1"] = record1_field

        if record2_field:
            fields["record2"] = record2_field

        if label_field:
            fields["label"] = label_field

        return Instance(fields) if fields or len(fields) > 0 else None


@DatasetReader.register("similarity_classifier")
class SimilarityClassifierDatasetReader(SequencePairClassifierDatasetReader):
    """A DatasetReader for the SimilarityClassifier model.
    """

    pass
