from allennlp.data import Token as AllennlpToken
from spacy.tokens.token import Token as SpacyToken

from biome.text.configuration import TokenizerConfiguration
from biome.text.tokenizer import Tokenizer

html_text = """
        <!DOCTYPE html>
        <html>
        <body>

        <h1>My First Heading</h1>
        <p>My first paragraph.</p>
        <p>My second paragraph.</p>
        </body>
        </html>
    """


def test_text_cleaning_with_sentence_segmentation():
    tokenizer = Tokenizer(
        TokenizerConfiguration(
            text_cleaning={"rules": ["html_to_text", "strip_spaces"]},
            segment_sentences=True,
        )
    )

    tokenized = tokenizer.tokenize_text(html_text)
    assert len(tokenized) == 2
    assert (
        len(tokenized[0]) == 7
    ), "Expected [My, First, Heading, My, first, paragraph, .]"
    assert len(tokenized[1]) == 4, "Expected [My, second, paragraph, .]"


def test_text_cleaning_with_sentence_segmentation_and_max_sequence():
    tokenizer = Tokenizer(
        TokenizerConfiguration(
            max_sequence_length=8,
            text_cleaning={"rules": ["html_to_text", "strip_spaces"]},
            segment_sentences=True,
        )
    )

    tokenized = tokenizer.tokenize_text(html_text)
    assert len(tokenized) == 2
    assert len(tokenized[0]) == 2, "Expected [My, First]"
    assert len(tokenized[1]) == 2, "Expected [My, second]"


def test_document_cleaning():
    tokenizer = Tokenizer(
        TokenizerConfiguration(
            text_cleaning={"rules": ["html_to_text", "strip_spaces"]},
            segment_sentences=True,
        )
    )

    tokenized = tokenizer.tokenize_document([html_text])
    assert len(tokenized) == 2
    assert (
        len(tokenized[0]) == 7
    ), "Expected [My, First, Heading, My, first, paragraph, .]"
    assert len(tokenized[1]) == 4, "Expected [My, second, paragraph, .]"


def test_using_spacy_tokens():
    tokenizer = Tokenizer(TokenizerConfiguration(use_spacy_tokens=True))
    tokenized = tokenizer.tokenize_text("This is a text")
    assert len(tokenized) == 1
    assert len(tokenized[0]) == 4
    assert all(map(lambda t: isinstance(t, SpacyToken), tokenized[0]))


def test_using_allennlp_tokens():
    tokenizer = Tokenizer(TokenizerConfiguration(use_spacy_tokens=False))
    tokenized = tokenizer.tokenize_text("This is a text")
    assert len(tokenized) == 1
    assert len(tokenized[0]) == 4
    assert all(map(lambda t: isinstance(t, AllennlpToken), tokenized[0]))


def test_set_sentence_segmentation_with_max_number_of_sentences():
    tokenizer = Tokenizer(TokenizerConfiguration(max_nr_of_sentences=2))
    tokenized = tokenizer.tokenize_document(
        [
            "This is a sentence. This is another sentence.",
            "One more sentence here.",
            "Last sentence here.",
        ]
    )
    assert len(tokenized) == 2
