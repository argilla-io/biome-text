import html
import re
from typing import List

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
        self._rules = rules or self.DEFAULT_RULES

    def __call__(self, text: str) -> str:
        for rule in self._rules:
            text = getattr(self, rule)(text)

        return text


@TextTransforms.register("rm_spaces")
class RmSpacesTextTransforms(TextTransforms):
    DEFAULT_RULES = TextTransforms.DEFAULT_RULES + [
        "strip_spaces",
        "rm_useless_spaces",
    ]

    @staticmethod
    def strip_spaces(text: str) -> str:
        """Strip leading and trailing spaces"""
        return text.strip()

    @staticmethod
    def rm_useless_spaces(text: str) -> str:
        """Remove multiple spaces in `text`"""
        return re.sub(" {2,}", " ", text)


@TextTransforms.register("fix_html")
class FixHtmlTextTransforms(RmSpacesTextTransforms):
    DEFAULT_RULES = RmSpacesTextTransforms.DEFAULT_RULES + ["fix_html"]

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
