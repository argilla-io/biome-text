import copy
import re
from typing import Callable
from typing import Dict
from typing import List

from allennlp.common import Registrable
from bs4 import BeautifulSoup


class TextCleaning(Registrable):
    """Defines rules that can be applied to the text before it gets tokenized.

    Each rule is a simple python function that receives and returns a `str`.

    Parameters
    ----------
    rules: `List[str]`
        A list of registered rule method names to be applied to text inputs
    """

    default_implementation = "default"

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


TextCleaning.register(TextCleaning.default_implementation)(TextCleaning)


class TextCleaningRule:
    """Registers a function as a rule for the text cleaning implementation

    Use the decorator `@TextCleaningRule` for creating custom text cleaning and pre-processing rules.

    An example function to strip spaces would be:

    ```python
    @TextCleaningRule
    def strip_spaces(text: str) -> str:
        return text.strip()
    ```

    You can query available rules via `TextCleaningRule.registered_rules()`.

    Parameters
    ----------
    func: `Callable[[str]`
        The function to register
    """

    __REGISTERED_RULES = {}

    def __init__(self, func: Callable[[str], str]):
        self.__callable__ = func
        self.__REGISTERED_RULES[func.__name__] = func

    @classmethod
    def registered_rules(cls) -> Dict[str, Callable[[str], str]]:
        """Registered rules dictionary"""
        return copy.deepcopy(cls.__REGISTERED_RULES)

    def __call__(self, *args, **kwargs) -> str:
        """Enables call single rule"""
        return self.__callable__(*args, **kwargs)


@TextCleaningRule
def strip_spaces(text: str) -> str:
    """Strips leading and trailing spaces/new lines"""
    return text.strip()


@TextCleaningRule
def rm_useless_spaces(text: str) -> str:
    """Removes multiple spaces in `str`"""
    return re.sub(" {2,}", " ", text)


@TextCleaningRule
def fix_html(text: str) -> str:
    """Replaces some special HTML characters: `&nbsp;`, `<br>`, etc."""
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
    """Extracts text from an HTML document"""
    return BeautifulSoup(text, "lxml").get_text()
