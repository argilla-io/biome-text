import inspect
import os
import os.path
import re
from inspect import Parameter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
import yaml
from allennlp.data import TextFieldTensors
from elasticsearch import Elasticsearch

from . import environment
from .features import CharFeatures, WordFeatures

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


def get_compatible_doc_type(client: Elasticsearch) -> str:
    """Find a compatible name for doc type by checking the cluster info

    Parameters
    ----------
    client
        The elasticsearch client

    Returns
    -------
    name
        A compatible name for doc type in function of cluster version
    """
    es_version = int(client.info()["version"]["number"].split(".")[0])
    return "_doc" if es_version >= 6 else "doc"


def get_env_cuda_device() -> int:
    """Gets the cuda device from an environment variable.

    This is necessary to activate a GPU if available

    Returns
    -------
    cuda_device
        The integer number of the CUDA device
    """
    cuda_device = int(os.getenv(environment.CUDA_DEVICE, "-1"))

    return cuda_device


def update_method_signature(signature: inspect.Signature, to_method: Callable) -> Callable:
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


def clean_metric_name(name: str) -> str:
    if not name:
        return name
    new_name = _INVALID_TAG_CHARACTERS.sub("_", name)
    new_name = new_name.lstrip("/")
    return new_name


def get_word_tokens_ids_from_text_field_tensors(
    text_field_tensors: TextFieldTensors,
) -> Optional[torch.Tensor]:
    """
    Given a text field tensor structure, tries to extract word features related tensors

    Parameters
    ----------
    text_field_tensors
        The incoming record text field tensors dictionary

    Returns
    -------
    tensor
        `WordFeatures` related tensors if enable
    """
    word_features_tensors = text_field_tensors.get(WordFeatures.namespace)
    if not word_features_tensors:
        return None

    for argument_name, tensor in word_features_tensors.items():
        if argument_name in ["tokens", "token_ids", "input_ids"]:
            return tensor


def get_char_tokens_ids_from_text_field_tensors(
    text_field_tensors: TextFieldTensors,
) -> Optional[torch.Tensor]:
    """
    Given a text field tensor structure, tries to extract character features related tensors

    See `TokenCharactersIndexer.tokens_to_indices` for more info

    Parameters
    ----------
    text_field_tensors
        The incoming record text field tensors dictionary

    Returns
    -------
    tensor
        `CharFeatures` related tensors if enable
    """
    char_features_tensors = text_field_tensors.get(CharFeatures.namespace)
    if not char_features_tensors:
        return None

    for argument_name, tensor in char_features_tensors.items():
        if argument_name in ["token_characters"]:
            return tensor


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
    """Creates an equivalent data structure representing data values as string"""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {key: stringify(value) for key, value in value.items()}
    if isinstance(value, Iterable):
        return [stringify(v) for v in value]
    return str(value)
