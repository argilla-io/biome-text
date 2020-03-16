from allennlp.common import Params
from biome.text.dataset_readers.text_transforms import TextTransforms, RmSpacesTransforms, Html2TextTransforms
import pytest


def test_text_transforms():
    params = Params({})
    text_transforms = TextTransforms.from_params(params)

    # assert default implementation, comes first
    assert type(text_transforms) == RmSpacesTransforms

    # assert return value
    assert isinstance(text_transforms("test"), str)

    # assert error if rule not found
    with pytest.raises(AttributeError):
        TextTransforms(rules="not_existing_rule")


def test_rm_spaces():
    text_transforms = TextTransforms.by_name("rm_spaces")()
    assert type(text_transforms) == RmSpacesTransforms
    assert text_transforms("\n\n  test this  text!\n  \n  ") == "test this text!"


def test_html_to_text():
    text_transforms = TextTransforms.by_name("html_to_text")()
    assert type(text_transforms) == Html2TextTransforms
    html_doc = """
    <html>
        <body>
            <p>    Hello @ViewBag.PersonName,</p>
            <p>This is a message &amp; text  </p>
        </body>
    </html>
    """
    assert text_transforms(html_doc) == "Hello @ViewBag.PersonName,\nThis is a message & text"




