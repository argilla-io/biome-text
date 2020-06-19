from biome.text.text_cleaning import DefaultTextCleaning
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
