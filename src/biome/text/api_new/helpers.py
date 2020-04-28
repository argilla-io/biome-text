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


def method_argument_types(the_method: Callable) -> Dict[str, Type[Any]]:
    """Returns the argument types dictionary"""

    def annotation_variants(annotation: Type) -> Set[Type]:
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if origin == Union:
            return set([t for t in args if t])
        return set([annotation])

    signature = inspect.signature(the_method)

    return {
        argument.name: variant
        for argument in signature.parameters.values()
        for variant in annotation_variants(remove_optional(argument.annotation))
    }


def update_method_signature(signature: inspect.Signature, to_method):
    """Updates signature to method"""

    def wrapper(*args, **kwargs):
        return to_method(*args, **kwargs)

    wrapper.__signature__ = signature
    return wrapper


def isgeneric(class_type: Type) -> bool:
    """Checks if a class type is a generic type (List[str] or Union[str, int]"""
    return hasattr(class_type, "__origin__")


def component_from_config(
    component_class: Type[Registrable],
    component_config: Dict[str, Any],
    previous: Optional[Registrable],
    **extra_args,
) -> Registrable:
    """Loads a allennlp registrable component class from its configuration"""

    init_args = method_argument_types(component_class.__init__)
    for arg_name, arg_type in init_args.items():
        if not isgeneric(arg_type) and isinstance(previous, arg_type):
            extra_args[arg_name] = previous
            break
    try:
        return component_class.from_params(
            Params(copy.deepcopy(component_config)), **extra_args
        )
    except:
        return component_config


def chain_component_config(
    component_class: Type[Registrable],
    component_config: Dict[str, Any],
    prev: Optional[Registrable],
) -> Dict[str, Any]:
    """
    Configures component forward chain by setting the component
    input dimension with previous output dimension
    """
    if not prev:
        return component_config

    if (
        component_config
        and isinstance(component_config, dict)
        and component_config.get("type")
        and hasattr(component_class, "by_name")
    ):
        component_class = component_class.by_name(component_config.get("type"))

    # This occurs with wrapped seq2vec/seq2seq and internal allennlp mechanism, the
    # Seq2VecWrapper/Seq2SeqWrapper class class. They break their own method api signature.
    # Bravo allennlp team !!
    if not isinstance(component_class, Type):
        component_class = component_class.__class__

    # There is no standardization about input dimension init field, so we need check
    # depending on the component class
    input_dim_attribute = None
    if issubclass(
        component_class,
        (
            _Seq2VecWrapper,
            _Seq2SeqWrapper,
            PytorchSeq2SeqWrapper,
            PytorchSeq2VecWrapper,
        ),
    ):
        input_dim_attribute = "input_size"
    else:
        init_method_keys = inspect.signature(component_class.__init__).parameters.keys()
        for param_name in ["embedding_dim", "input_dim"]:
            if param_name in init_method_keys:
                input_dim_attribute = param_name
                break

    if not input_dim_attribute:
        return component_config

    if hasattr(prev, "get_output_dim"):
        return {input_dim_attribute: prev.get_output_dim(), **component_config}
    raise TypeError(f"Cannot chain from component {prev}")


class WithLayerChain(Registrable):
    """TBD"""

    @classmethod
    def from_params(cls: Type[Registrable], params: Params, **extras) -> Any:
        the_class = cls
        if params.get("type") and hasattr(cls, "by_name"):
            the_class = the_class.by_name(params.pop("type"))

        args = {}
        init_model_signature = inspect.signature(the_class.__init__)
        prev_component = extras.get("prev_layer")  # This is a must
        for name, component_cfg in params.as_dict().items():
            parameter = init_model_signature.parameters.get(name)
            if not parameter:
                args[name] = component_cfg
                continue

            component_class = remove_optional(parameter.annotation)
            new_configuration = chain_component_config(
                component_class, component_cfg, prev_component
            )
            prev_component = component_from_config(
                component_class, new_configuration, previous=prev_component, **extras
            )
            args[name] = prev_component

        return the_class(**args, **extras)


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
