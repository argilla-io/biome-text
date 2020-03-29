import inspect
from typing import Any, Generic, TypeVar, Type, Optional, Dict

from allennlp.common import Params, FromParams
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper

from biome.text.pipelines._impl.allennlp.models.defs import ChainedLayer


def _find_input_attribute(component: Any) -> str:
    """Find the properly input dimension attribute name for a given component"""
    input_dim_attribute = None
    if isinstance(component, (_Seq2SeqWrapper, _Seq2VecWrapper)):
        input_dim_attribute = "input_size"
    else:
        init_method_keys = inspect.signature(component.__init__).parameters.keys()
        for param_name in ["embedding_dim", "input_dim"]:
            if param_name in init_method_keys:
                input_dim_attribute = param_name
                break
    return input_dim_attribute


T = TypeVar("T")


class LayerSpec(Generic[T], FromParams):
    """
    The layer spec component allows create
    Pytorch modules lazily, and instantiate its components with dimension layer chain.

    The layer spec wraps a component params and will generate an instance of type T once the input_dim is set.

    """

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        return cls(config=params.as_dict())

    @classmethod
    def __resolve_layer_class(cls) -> Type[T]:
        return getattr(cls, "__orig_bases__")[0].__args__[0]

    def __init__(self, config=None):
        self._config = config
        self._input_dim = None

    def input_dim(self, input_dim: int) -> "LayerSpec":
        """Sets the input dimension attribute for this layer configuration"""
        self._input_dim = input_dim
        return self

    def compile(self, **extras) -> T:
        """
        Using the wrapped configuration and the input dimension, generates a
        instance of type T representing the layer configuration
        """
        if not self._config:
            return None

        layer_class = self.__resolve_layer_class()
        component = layer_class.by_name(self._config.get("type"))
        input_dim_attribute = _find_input_attribute(component)
        if input_dim_attribute:
            if not self._input_dim:
                raise ValueError(
                    "Must set the input dimension before compile configuration"
                )
            self._config[input_dim_attribute] = self._input_dim

        return layer_class.from_params(Params(self._config), **extras)


class Seq2VecEncoderSpec(LayerSpec[Seq2VecEncoder]):
    """Layer spec for Seq2VecEncoder components"""

    pass


class Seq2SeqEncoderSpec(LayerSpec[Seq2SeqEncoder]):
    """Layer spec for Seq2SeqEncoder components"""

    pass


class FeedForwardSpec(LayerSpec[FeedForward]):
    """Layer spec for FeedForward components"""

    pass
