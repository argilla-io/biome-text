import copy
import datetime
import inspect
import os
from inspect import Parameter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import yaml
from allennlp.common import Params, Registrable
from allennlp.common.from_params import remove_optional
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, _Seq2VecWrapper
from elasticsearch import Elasticsearch

from biome.text.api_new import environment


def yaml_to_dict(filepath: str) -> Dict[str, Any]:
    """Loads a yaml file into a data dictionary"""
    with open(filepath) as yaml_content:
        config = yaml.safe_load(yaml_content)
    return config


def get_compatible_doc_type(client: Elasticsearch) -> str:
    """
    Find a compatible name for doc type by checking the cluster info
    Parameters
    ----------
    client
        The elasticsearch client

    Returns
    -------
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


def update_method_signature(signature: inspect.Signature, to_method):
    """Updates signature to method"""

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
