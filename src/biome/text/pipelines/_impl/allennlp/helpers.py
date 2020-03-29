import copy
import inspect
from typing import Type, Dict, Any, Optional, Callable, List

from allennlp.common import Registrable, Params
from allennlp.common.from_params import remove_optional
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper, PytorchSeq2VecWrapper


def method_argument_types(the_method: Callable) -> Dict[str, Type[Any]]:
    """Returns the argument types dictionary"""
    signature = inspect.signature(the_method)

    return {
        argument.name: remove_optional(argument.annotation)
        for argument in signature.parameters.values()
    }


def component_from_config(
    component_class: Type[Registrable],
    component_config: Dict[str, Any],
    previous: Optional[Registrable],
    **extra_args,
) -> Registrable:
    """Loads a allennlp registrable component class from its configuration"""
    init_args = method_argument_types(component_class.__init__)
    for arg_name, arg_type in init_args.items():
        if isinstance(previous, arg_type):
            extra_args[arg_name] = previous
            break
    return component_class.from_params(
        Params(copy.deepcopy(component_config)), **extra_args
    )


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

    if component_config.get("type") and hasattr(component_class, "by_name"):
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
