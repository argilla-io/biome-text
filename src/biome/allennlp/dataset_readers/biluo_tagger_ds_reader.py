import glob
import logging
from typing import List, Dict, Iterable, Optional, Any

import pandas
import spacy
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Token, Instance, Field, TokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from spacy.gold import biluo_tags_from_offsets

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
        format: str = "biluo",
        lang: str = None,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_ns = label_namespace
        self._format = format.lower()
        self._nlp = None
        self._lang = lang or "en"

    def _read(self, file_path: str) -> Iterable[Instance]:
        for file in glob.glob(file_path):
            df = pandas.read_json(cached_path(file), orient="records", lines=True)

            if self._format == "biluo":
                tokens_tags = zip(df.tokens.values, df.tags.values)

            elif self._format == "spans":
                tokens_tags = [
                    data.values()
                    for data in self._docs_2_biluo(
                        df.to_dict(orient="records"), lang=self._lang
                    )
                ]

            for tokens, tags in tokens_tags:
                yield self.text_to_instance(
                    tokens=[Token(t) for t in tokens], ner_tags=tags
                )

    def _doc_2_biluo(self, data, doc) -> Optional[Dict[str, Any]]:
        try:
            entities = [
                (annotation["start"], annotation["end"], annotation["entity"])
                for annotation in data.get("annotations", [])
                if annotation.get("start")
                and annotation.get("end")
                and annotation.get("entity")
            ]

            tokens = [token.text for token in doc]
            return dict(tokens=tokens, tags=biluo_tags_from_offsets(doc, entities))
        except:
            return None

    def _spacy_load(self, name: str, **kwargs):
        try:
            if not self._nlp:
                self._nlp = spacy.load(name, **kwargs)
            return self._nlp
        except OSError:
            from spacy.cli import download

            download(name)

            return self._spacy_load(name, **kwargs)

    def _docs_2_biluo(
        self, documents: Iterable[Dict[str, Any]], lang: str = "en"
    ) -> Iterable[Dict[str, Any]]:
        nlp = self._spacy_load(lang)
        docs = list(nlp.pipe([document["text"] for document in documents]))

        return [
            biluo
            for biluo in (
                self._doc_2_biluo(data, doc) for data, doc in zip(documents, docs)
            )
            if biluo is not None
        ]

    def text_to_instance(
        self,  # type: ignore
        tokens: List[Token],
        ner_tags: List[str] = None,
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
            # "ner_tags": SequenceLabelField(ner_tags, sequence, "ner_tags"),
        }

        if ner_tags:
            instance_fields["tags"] = SequenceLabelField(
                ner_tags, sequence, self._label_ns
            )

        if pos_tags:
            instance_fields["pos_tags"] = SequenceLabelField(
                pos_tags, sequence, "pos_tags"
            )

        return Instance(instance_fields)
