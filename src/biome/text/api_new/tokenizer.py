from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from allennlp.common import FromParams
from allennlp.data import Token
from allennlp.data.tokenizers import SentenceSplitter, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from biome.text.api_new.text_cleaning import DefaultTextCleaning, TextCleaning


class Tokenizer(FromParams):
    """Manages data tokenization"""

    def __init__(
        self,
        lang: str = "en",
        skip_empty_tokens: bool = False,
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
        text_cleaning: Optional[TextCleaning] = None,
        segment_sentences: Union[bool, SentenceSplitter] = False,
        start_tokens: Optional[List[str]] = None,
        end_tokens: Optional[List[str]] = None,
    ):
        if segment_sentences is True:
            # TODO: check rule-based feat.
            segment_sentences = SpacySentenceSplitter(language=lang, rule_based=True)

        self.lang = lang
        self.segment_sentences = segment_sentences
        self.skip_empty_tokens = skip_empty_tokens
        self.max_nr_of_sentences = max_nr_of_sentences
        self.max_sequence_length = max_sequence_length
        self.text_cleaning = text_cleaning or DefaultTextCleaning()

        self._base_tokenizer = WordTokenizer(
            word_splitter=SpacyWordSplitter(language=self.lang),
            start_tokens=start_tokens,
            end_tokens=end_tokens,
        )

    def _text_cleaning(self, text) -> str:
        return self.text_cleaning(text)

    def _value_as_string(self, value: Any) -> str:
        """Converts a value data into its string representation"""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return self._value_as_string(value.values())
        if isinstance(value, Iterable):
            return " ".join(map(self._value_as_string, value))
        return str(value)

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Transforms an data dictionary into a dictionary of strings values"""
        return {key: self._value_as_string(value) for key, value in data.items()}

    def tokenize_record(
        self, record: Dict[str, Any]
    ) -> Dict[str, Tuple[List[Token], List[Token]]]:
        """
        Tokenize a record data and generates a new data dictionary with two
        list of tokens for each record entry: key and value tokens
        """
        data = self._sanitize_dict(record)
        return {
            key: (self.tokenize_text(key), self.tokenize_text(value))
            for key, value in data.items()
        }

    def tokenize_document(self, document: List[str]) -> List[List[Token]]:
        """
        Tokenize a list of text (document).
        The resultant length list could differs if segment sentences flag is enabled
        """

        sentences = document
        if self.segment_sentences:
            sentences = [
                sentence
                for sentences in self.segment_sentences.batch_split_sentences(document)
                for sentence in sentences
            ]
        return [
            self.tokenize_text(sentence)
            for sentence in sentences[: self.max_nr_of_sentences]
        ]

    def tokenize_text(self, text: str) -> List[Token]:
        """Tokenize a text"""
        return self._base_tokenizer.tokenize(
            self._text_cleaning(text)[: self.max_sequence_length]
        )
