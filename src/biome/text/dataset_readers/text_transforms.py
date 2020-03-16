import html
import re
from typing import List
from bs4 import BeautifulSoup

from allennlp.common import Registrable


class TextTransforms(Registrable):
    """This class defines some rules that can be applied to the text before it gets embedded in a `TextField`.

    Each rule is a simple python class method that receives and returns a str.
    It will be applied when an instance of this class is called.

    Parameters
    ----------
    rules
        A list of class method names to be applied on calling the instance.

    Attributes
    ----------
    DEFAULT_RULES
        The default rules if the `rules` parameter is not provided.
    """

    default_implementation = "rm_spaces"

    DEFAULT_RULES = []

    def __init__(self, rules: List[str] = None):
        self.rules = rules or self.DEFAULT_RULES
        for rule in self.rules:
            if not hasattr(self, rule):
                raise AttributeError(f"{type(self).__name__} has no rule (method) called '{rule}'")

    def __call__(self, text: str) -> str:
        for rule in self.rules:
            text = getattr(self, rule)(text)

        return text


@TextTransforms.register("rm_spaces")
class RmSpacesTransforms(TextTransforms):
    DEFAULT_RULES = TextTransforms.DEFAULT_RULES + [
        "strip_spaces",
        "rm_useless_spaces",
    ]

    @staticmethod
    def strip_spaces(text: str) -> str:
        """Strip leading and trailing spaces/new lines"""
        return text.strip()

    @staticmethod
    def rm_useless_spaces(text: str) -> str:
        """Remove multiple spaces in `text`"""
        return re.sub(" {2,}", " ", text)


@TextTransforms.register("html_to_text")
class Html2TextTransforms(RmSpacesTransforms):
    DEFAULT_RULES = ["html_to_text", "fix_html"] + RmSpacesTransforms.DEFAULT_RULES

    @staticmethod
    def html_to_text(text: str) -> str:
        """Extract text from a html doc with BeautifulSoup4"""
        return BeautifulSoup(text, "lxml").get_text()

    @staticmethod
    def fix_html(text: str) -> str:
        """List of replacements from html strings in `text`.
        Copied from https://docs.fast.ai/text.transform.html#fix_html"""
        re1 = re.compile(r"  +")
        text = (
            text.replace("#39;", "'")
            .replace("amp;", "&")
            .replace("#146;", "'")
            .replace("nbsp;", " ")
            .replace("#36;", "$")
            .replace("\\n", "\n")
            .replace("quot;", "'")
            .replace("<br />", "\n")
            .replace('\\"', '"')
            .replace("<unk>", "xxunk")
            .replace(" @.@ ", ".")
            .replace(" @-@ ", "-")
            .replace(" @,@ ", ",")
            .replace("\\", " \\ ")
        )
        return re1.sub(" ", html.unescape(text))
