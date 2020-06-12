from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from allennlp.common import FromParams
from allennlp.data import Token
from allennlp.data.tokenizers import SentenceSplitter, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

from biome.text.text_cleaning import DefaultTextCleaning, TextCleaning


class Tokenizer(FromParams):
    """Pre-processes and tokenizes the input text
    
    Transforms inputs (e.g., a text, a list of texts, etc.) into structures containing `allennlp.data.Token` objects.
    
    Use its arguments to configure the first stage of the pipeline (i.e., pre-processing a given set of text inputs.)
    
    Use methods for tokenization depending on the shape of the inputs
    (e.g., records with multiple fields, sentences lists).
    
    Parameters
    ----------
    lang
        The [spaCy model used](https://spacy.io/api/tokenizer) for tokenization is language dependent.
        For optimal performance, specify the language of your input data (default: "en").
    max_sequence_length: `int`
        Maximum length in characters for input texts truncated with `[:max_sequence_length]` after `TextCleaning`.
    max_nr_of_sentences: `int`
        Maximum number of sentences to keep when using `segment_sentences` truncated with `[:max_sequence_length]`.
    text_cleaning: `Optional[TextCleaning]`
        A `TextCleaning` configuration with pre-processing rules for cleaning up and transforming raw input text.
    segment_sentences:  `Union[bool, SentenceSplitter]`
        Whether to segment input texts in to sentences using the default `SentenceSplitter` or a given splitter.
    start_tokens: `Optional[List[str]]`
        A list of token strings to the sequence before tokenized input text.
    end_tokens: `Optional[List[str]]`
        A list of token strings to the sequence after tokenized input text.
    """

    # TODO: Check if it is possible to pass in a `TokenizerConfiguration` to avoid duplication of arguments

    def __init__(
        self,
        lang: str = "en",
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
        text_cleaning: Optional[TextCleaning] = None,
        segment_sentences: Union[bool, SentenceSplitter] = False,
        start_tokens: Optional[List[str]] = None,
        end_tokens: Optional[List[str]] = None,
    ):
        self._fetch_spacy_model(lang)

        if segment_sentences is True:
            # TODO: check rule-based feat.
            segment_sentences = SpacySentenceSplitter(language=lang, rule_based=True)

        self.lang = lang
        self.segment_sentences = segment_sentences
        self.max_nr_of_sentences = max_nr_of_sentences
        self.max_sequence_length = max_sequence_length
        self.text_cleaning = text_cleaning or DefaultTextCleaning()

        self._base_tokenizer = SpacyTokenizer(
            language=self.lang, start_tokens=start_tokens, end_tokens=end_tokens,
        )

    def _fetch_spacy_model(self, lang):
        # Allennlp get_spacy_model method works only for fully named models (en_core_web_sm) but no
        # for already linked named (en, es)
        # This is a workaround for mitigate those kind of errors. Just loading one more time, it's ok.
        # See https://github.com/allenai/allennlp/issues/4201
        import spacy

        try:
            spacy.load(lang, disable=["vectors", "textcat", "tagger" "parser" "ner"])
        except OSError:
            spacy.cli.download(lang)

    def tokenize_text(self, text: str) -> List[List[Token]]:
        """
        Tokenizes a text string applying sentence segmentation, if enabled

        Parameters
        ----------
        text: `str`
            The input text

        Returns
        -------
        A list of list of `Token`.

        If no sentence segmentation is enabled, or just one sentence is found in text
        the first level list will contain just one element: the tokenized text.

        """
        return self.tokenize_document([text])

    def _tokenize(self, text: str) -> List[Token]:
        """Tokenizes an input text string

        The simplest case where your input is just a `str`. For a text tokenization with
        sentence segmentation, see `tokenize_text`


        Parameters
        ----------
        text: `str`
            The input text
        Returns
        -------
        tokens: `List[Token]`

        """
        return self._base_tokenizer.tokenize(text[: self.max_sequence_length])

    def tokenize_document(self, document: List[str]) -> List[List[Token]]:
        """ Tokenizes a document-like structure containing lists of text inputs

        Use this to account for hierarchical text structures (e.g., a paragraph is a list of sentences)

        Parameters
        ----------
        document: `List[str]`
            A `List` with text inputs, e.g., sentences
            
        Returns
        -------
        tokens: `List[List[Token]]`
        """
        sentences = [self.text_cleaning(text) for text in document]
        if self.segment_sentences:
            sentences = [
                sentence
                for sentences in self.segment_sentences.batch_split_sentences(sentences)
                for sentence in sentences
            ]
        return [
            self._tokenize(sentence)
            for sentence in sentences[: self.max_nr_of_sentences]
        ]

    def tokenize_record(
        self, record: Dict[str, Any], exclude_record_keys: bool
    ) -> List[List[Token]]:
        """ Tokenizes a record-like structure containing text inputs
        
        Use this to keep information about the record-like data structure as input features to the model.
        
        Parameters
        ----------
        record: `Dict[str, Any]`
            A `Dict` with arbitrary "fields" containing text.
        exclude_record_keys: `bool`
            If enabled, exclude tokens related to record key text
            
        Returns
        -------
        tokens: `List[List[Token]]`
            A list of tokenized fields as token list
        """
        data = self._sanitize_dict(record)
        if exclude_record_keys:
            return [
                sentence
                for key, value in data.items()
                for sentence in self.tokenize_text(value)
            ]

        return [
            tokenized_key + sentence
            for key, value in data.items()
            for tokenized_key in [self._tokenize(key)]
            for sentence in self.tokenize_text(value)
        ]

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
