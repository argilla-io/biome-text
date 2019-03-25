import glob
import logging
from typing import List, Dict, Iterable

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Token, Instance, Field, TokenIndexer
from allennlp.data.fields import TextField, MetadataField, SequenceLabelField
import pandas
from allennlp.data.token_indexers import SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sequence-tagger")
class BiluoSequenceTaggerDatasetReader(DatasetReader):
    """
    Read instances from a set of json-based where every line is a json object with
    two fields:

    "tokens": a list of tokenized sentence
    "tags": corresponding to biluo tag annotation for tokens

    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        label_namespace: str = "labels",
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_ns = label_namespace

    def _read(self, file_path: str) -> Iterable[Instance]:

        df = pandas.concat(
            [
                pandas.read_json(cached_path(file), orient="records", lines=True)
                for file in glob.glob(file_path)
            ]
        )

        for tokens, tags in df.values:
            yield self.text_to_instance(
                tokens=[Token(t) for t in tokens], ner_tags=tags
            )

    def text_to_instance(
        self,  # type: ignore
        tokens: List[Token],
        ner_tags: List[str],
        pos_tags: List[str] = None,
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {
            "tokens": sequence,
            "metadata": MetadataField({"words": [x.text for x in tokens]}),
            "ner_tags": SequenceLabelField(ner_tags, sequence, "ner_tags"),
            "tags": SequenceLabelField(ner_tags, sequence, self._label_ns),
        }

        if pos_tags:
            instance_fields["pos_tags"] = SequenceLabelField(
                pos_tags, sequence, "pos_tags"
            )

        return Instance(instance_fields)
