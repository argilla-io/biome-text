import copy
import functools
import inspect
import os
import os.path
import re
from inspect import Parameter
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import yaml
from allennlp.common import util
from allennlp.data import Token as AllenNLPToken
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from spacy.tokens import Token as SpacyToken
from spacy.tokens.doc import Doc
from spacy.training import biluo_tags_to_offsets
from spacy.training import offsets_to_biluo_tags

_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")


def yaml_to_dict(filepath: str) -> Dict[str, Any]:
    """Loads a yaml file into a data dictionary

    Parameters
    ----------
    filepath
        Path to the yaml file

    Returns
    -------
    dict
    """
    with open(filepath) as yaml_content:
        config = yaml.safe_load(yaml_content)
    return config


def update_method_signature(
    signature: inspect.Signature, to_method: Callable
) -> Callable:
    """Updates the signature of a method

    Parameters
    ----------
    signature
        The signature with which to update the method
    to_method
        The method whose signature will be updated

    Returns
    -------
    updated_method
    """

    def wrapper(*args, **kwargs):
        return to_method(*args, **kwargs)

    wrapper.__signature__ = signature
    wrapper.__doc__ = to_method.__doc__

    return wrapper


def isgeneric(class_type: Type) -> bool:
    """Checks if a class type is a generic type (List[str] or Union[str, int]"""
    return hasattr(class_type, "__origin__")


def is_running_on_notebook() -> bool:
    """Checks if code is running inside a jupyter notebook"""
    try:
        import IPython

        return IPython.get_ipython().has_trait("kernel")
    except (AttributeError, NameError, ModuleNotFoundError):
        return False


def split_signature_params_by_predicate(
    signature_function: Callable, predicate: Callable
) -> Tuple[List[Parameter], List[Parameter]]:
    """Splits parameters signature by defined boolean predicate function"""
    signature = inspect.signature(signature_function)
    parameters = list(
        filter(
            lambda p: p.name != "self"
            and p.kind not in [Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL],
            signature.parameters.values(),
        )
    )
    matches_group = list(filter(lambda p: predicate(p), parameters))
    non_matches_group = list(filter(lambda p: not predicate(p), parameters))

    return matches_group, non_matches_group


def sanitize_metric_name(name: str) -> str:
    """Sanitizes the name to comply with tensorboardX conventions when logging.

    Parameter
    ---------
    name
        Name of the metric

    Returns
    -------
    sanitized_name
    """
    if not name:
        return name
    new_name = _INVALID_TAG_CHARACTERS.sub("_", name)
    new_name = new_name.lstrip("/")
    return new_name


def save_dict_as_yaml(dictionary: dict, path: str) -> str:
    """Save a cfg dict to path as yaml

    Parameters
    ----------
    dictionary
        Dictionary to be saved
    path
        Filesystem location where the yaml file will be saved

    Returns
    -------
    path
        Location of the yaml file
    """
    dir_name = os.path.dirname(path)
    # Prevent current workdir relative routes
    # `save_dict_as_yaml("just_here.yml")
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w") as yml_file:
        yaml.dump(dictionary, yml_file, default_flow_style=False, allow_unicode=True)

    return path


def get_full_class_name(the_class: Type) -> str:
    """Given a type class return the full qualified class name """
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = the_class.__module__
    if module is None or module == str.__class__.__module__:
        return the_class.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + the_class.__name__


def stringify(value: Any) -> Any:
    """Creates an equivalent data structure representing data values as string

    Parameters
    ----------
    value
        Value to be stringified

    Returns
    -------
    stringified_value
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {key: stringify(value) for key, value in value.items()}
    if isinstance(value, Iterable):
        return [stringify(v) for v in value]
    return str(value)


def sanitize_for_params(x: Any) -> Any:
    """Sanitizes the input for a more flexible usage with AllenNLP's `.from_params()` machinery.

    For now it is mainly used to transform numpy numbers to python types

    Parameters
    ----------
    x
        The parameter passed on to `allennlp.common.FromParams.from_params()`

    Returns
    -------
    sanitized_x
    """
    # AllenNLP has a similar function (allennlp.common.util.sanitize) but it does not work for my purpose, since
    # numpy types are checked only after the float type check, and:
    # isinstance(numpy.float64(1), float) == True !!!
    if isinstance(x, util.numpy.number):
        return x.item()
    elif isinstance(x, util.numpy.bool_):
        # Numpy bool_ need to be converted to python bool.
        return bool(x)
    if isinstance(x, (str, float, int, bool)):
        return x
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize_for_params(value) for key, value in x.items()}
    # Lists and Tuples need their values sanitized
    elif isinstance(x, list):
        return [sanitize_for_params(x_i) for x_i in x]
    elif isinstance(x, tuple):
        return tuple(sanitize_for_params(x_i) for x_i in x)
    # We include `to_json` function customize sanitization for user defined classes
    elif hasattr(x, "to_json"):
        return x.to_json()
    return x


def sanitize_for_yaml(value: Any):
    """Sanitizes the value for a simple yaml output, that is classes only built-in types"""
    if isinstance(value, list):
        return [sanitize_for_yaml(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_yaml(v) for v in value)
    if isinstance(value, dict):
        return {k: sanitize_for_yaml(v) for k, v in value.items()}
    try:
        yaml.dump(value)
    # we try a string representation
    except Exception:
        return str(value)
    else:
        return value


def span_labels_to_tag_labels(
    labels: List[str], label_encoding: str = "BIO"
) -> List[str]:
    """Converts a list of span labels to tag labels following `spacy.training.offsets_to_biluo_tags`

    Parameters
    ----------
    labels
        Span labels to convert
    label_encoding
        The label format used for the tag labels

    Returns
    -------
    tag_labels
    """
    if label_encoding == "BIOUL":
        converted_labels = [
            f"{char}-{label}" for char in ["B", "I", "U", "L"] for label in labels
        ] + ["O"]
    elif label_encoding == "BIO":
        converted_labels = [
            f"{char}-{label}" for char in ["B", "I"] for label in labels
        ] + ["O"]
    else:
        raise ValueError(
            f"'{label_encoding}' is not a supported label encoding scheme."
        )

    return converted_labels


def bioul_tags_to_bio_tags(tags: List[str]) -> List[str]:
    """Converts BIOUL tags to BIO tags

    Parameters
    ----------
    tags
        BIOUL tags to convert

    Returns
    -------
    bio_tags
    """
    return [tag.replace("L-", "I-", 1).replace("U-", "B-", 1) for tag in tags]


def tags_from_offsets(
    doc: Doc,
    offsets: List[Dict],
    label_encoding: Optional[str] = "BIOUL",
) -> List[str]:
    """Converts offsets to BIOUL or BIO tags using spacy's `offsets_to_biluo_tags`.

    Parameters
    ----------
    doc
        A spaCy Doc created with `text` and the backbone tokenizer
    offsets
        A list of dicts with start and end character index with respect to the doc, and the span label:
        `{"start": int, "end": int, "label": str}`
    label_encoding
        The label encoding to be used: BIOUL or BIO

    Returns
    -------
    tags (BIOUL or BIO)
    """
    tags = offsets_to_biluo_tags(
        doc, [(offset["start"], offset["end"], offset["label"]) for offset in offsets]
    )
    if label_encoding == "BIO":
        tags = bioul_tags_to_bio_tags(tags)
    return tags


def offsets_from_tags(
    doc: Doc,
    tags: List[str],
    label_encoding: Optional[str] = "BIOUL",
    only_token_spans: bool = False,
) -> List[Dict]:
    """Converts BIOUL or BIO tags to offsets

    Parameters
    ----------
    doc
        A spaCy Doc created with `text` and the backbone tokenizer
    tags
        A list of BIOUL or BIO tags
    label_encoding
        The label encoding of the tags: BIOUL or BIO
    only_token_spans
        If True, offsets contains only token index references. Default is False

    Returns
    -------
    offsets
        A list of dicts with start and end character/token index with respect to the doc and the span label:
        `{"start": int, "end": int, "start_token": int, "end_token": int, "label": str}`
    """
    # spacy's biluo_tags_to_offsets surprisingly does not check this ...
    if len(doc) != len(tags):
        raise ValueError(
            f"Number of tokens and tags must be the same, "
            f"but 'len({list(doc)}) != len({tags})"
        )

    if label_encoding == "BIO":
        tags = to_bioul(tags, encoding="BIO")

    offsets = []
    for start, end, label in biluo_tags_to_offsets(doc, tags):
        span = doc.char_span(start, end)
        data = {
            "start_token": span.start,
            "end_token": span.end,
            "label": label,
        }
        if not only_token_spans:
            data.update({"start": start, "end": end})
        offsets.append(data)
    return offsets


def merge_dicts(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries recursivelly

    Examples
    --------
    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge_dicts(b, a)
    {'first': {'all_rows': {'pass': 'dog', 'number': '5', 'fail': 'cat'}}}

    """
    if not isinstance(destination, dict):
        return source

    result = copy.deepcopy(destination)

    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = result.setdefault(key, {})
            value = merge_dicts(value, node)
        result[key] = value

    return result


def copy_sign_and_docs(org_func):
    """Decorator to copy the signature and the docstring from the org_func"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(org_func)
        wrapper.__doc__ = org_func.__doc__

        return wrapper

    return decorator


def spacy_to_allennlp_token(token: SpacyToken) -> AllenNLPToken:
    return AllenNLPToken(
        text=token.text,
        idx=token.idx,
        idx_end=token.idx + len(token),
        lemma_=token.lemma_,
        pos_=token.pos_,
        tag_=token.tag_,
        dep_=token.dep_,
        ent_type_=token.ent_type_,
    )
