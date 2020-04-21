import copy
import re
from typing import Callable, Dict, List

from allennlp.common import Registrable
from bs4 import BeautifulSoup


class TextCleaning(Registrable):
    """
    Base class for text cleaning
    """

    default_implementation = "default"

    def __call__(self, text: str) -> str:
        """Apply the text transformation"""
        raise NotImplementedError


class TextCleaningRule:
    """This decorator allows register a function as an available rule for the default text cleaning implementation"""

    __REGISTERED_RULES = {}

    def __init__(self, func: Callable[[str], str]):
        self.__REGISTERED_RULES[func.__name__] = func

    @classmethod
    def registered_rules(cls) -> Dict[str, Callable[[str], str]]:
        """Registered rules dictionary"""
        return copy.deepcopy(cls.__REGISTERED_RULES)


@TextCleaning.register(TextCleaning.default_implementation)
class DefaultTextCleaning(Registrable):
    """
    This class defines some rules that can be applied to the text before it gets embedded in a `TextField`.

    Each rule is a simple python function that receives and returns a str.


    Parameters
    ----------
    rules
        A list of registered rule method names to be applied on calling the instance.

    Attributes
    ----------
        The default rules if the `rules` parameter is not provided.
    """

    def __init__(self, rules: List[str] = None):
        self.rules = rules or []
        for rule in self.rules:
            if rule not in TextCleaningRule.registered_rules():
                raise AttributeError(
                    f"No rule '{rule}' registered"
                    f"Available rules are [{[k for k in TextCleaningRule.registered_rules().keys()]}]"
                )

    def __call__(self, text: str) -> str:
        for rule in self.rules:
            text = TextCleaningRule.registered_rules()[rule](text)
        return text


@TextCleaningRule
def strip_spaces(text: str) -> str:
    """Strip leading and trailing spaces/new lines"""
    return text.strip()


@TextCleaningRule
def rm_useless_spaces(text: str) -> str:
    """Remove multiple spaces in `text`"""
    return re.sub(" {2,}", " ", text)


@TextCleaningRule
def fix_html(text: str) -> str:
    """list of replacements in html code.
    I leave a link to the fastai version here as a reference:
    https://docs.fast.ai/text.transform.html#fix_html
    """
    text = (
        # non breakable space -> space
        text.replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&#xa0;", " ")
        # <br> html single line breaks -> unicode line breaks
        .replace("<br>", "\n")
    )

    return text


@TextCleaningRule
def html_to_text(text: str) -> str:
    """Extract text from a html doc with BeautifulSoup4"""
    return BeautifulSoup(text, "lxml").get_text()
