import logging

from allennlp.common import Params
from allennlp.data import Token
from overrides import overrides
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.common.util import get_spacy_model
from typing import List

_logger = logging.getLogger(__name__)


class SpacyWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
    recommended ``WordSplitter``.
    """

    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # Return the real token type
        return [Token(t.text,
                      t.idx,
                      t.pos,
                      t.tag,
                      t.dep,
                      t.ent_type)
                for t in self.spacy(sentence) if not t.is_space]
