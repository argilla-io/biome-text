from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from allennlp.common import FromParams
from allennlp.data import Token
from allennlp.data.tokenizers import SentenceSplitter, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from biome.text.api_new.text_cleaning import DefaultTextCleaning, TextCleaning


class Tokenizer(FromParams):
    """Pre-processes and tokenizes input text
    
    Transforms inputs (e.g., a text, a list of texts, etc.) into structures containing `allennlp.data.Token` objects.
    
    Use its arguments to configure the first stage of the pipeline (i.e., pre-processing a given set of text inputs.)
    
    Use methods for tokenizing depending on the shape of inputs (e.g., records with multiple fields, sentences lists).
    
    The main usage of its methods is for developing custom `TaskHead` implementations.
    
    # Parameters
        lang: `str`
            The `spaCy` language to be used by the tokenizer (default is `en`)
        skip_empty_tokens: `bool`
        max_sequence_length: `int`
            Maximum length in characters for input texts.
            Texts will be truncated with `[:max_sequence_length]` after the `TextCleaning` stage.
        max_nr_of_sentences: `int`
            Maximum number of sentences to keep when using `segment_sentences`.
            The list of sentences will be truncated  with `[:max_sequence_length]`.
        text_cleaning: `Optional[TextCleaning]`
            A `TextCleaning` configuration with pre-processing rules for cleaning up and transforming raw input text.
        segment_sentences:  `Union[bool, SentenceSplitter]`
            Whether to segment input texts in to sentences using the default `SentenceSplitter` or a given splitter.
        start_tokens: `Optional[List[str]]`
            A list of token strings to the sequence before tokenized input text.
        end_tokens: `Optional[List[str]]`
            A list of token strings to the sequence after tokenized input text.
    """

    def __init__(
        self,
        lang: str = "en",
        skip_empty_tokens: bool = False,  # TODO(David): Check this param for the new API
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

    def tokenize_text(self, text: str) -> List[Token]:
        """ Tokenizes a text string

        Use this for the simplest case where your input is just a `str`

        # Parameters
            text: `str`
        # Returns`
            A `List[Token]` object
        """
        return self._base_tokenizer.tokenize(
            self._text_cleaning(text)[: self.max_sequence_length]
        )

    def tokenize_document(self, document: List[str]) -> List[List[Token]]:
        """ Tokenizes a document-like structure containing lists of text inputs

        Use this to account for hierarchical text structures (e.g., a paragraph is a list of sentences)

        # Parameters
            document: `List[str]`
            A `List` with text inputs, e.g., sentences
        # Returns`
            A list of lists of tokens `List[List[Token]]`
        """
        """
        TODO: clarify?: The resultant length list could differs if segment sentences flag is enabled
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

    def tokenize_record(
        self, record: Dict[str, Any]
    ) -> Dict[str, Tuple[List[Token], List[Token]]]:
        """ Tokenizes a record-like structure containing text inputs
        
        Use this to keep information about the record-like data structure as input features to the model.
        
        # Parameters
            record: `Dict[str, Any]`
            A `Dict` with arbitrary "fields" containing text.
        # Returns
            A data dictionary with two lists of `Token`'s for each record entry: `key` and `value` tokens.
            Type: `Dict[str, Tuple[List[Token], List[Token]]]`
        """
        data = self._sanitize_dict(record)
        return {
            key: (self.tokenize_text(key), self.tokenize_text(value))
            for key, value in data.items()
        }

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
