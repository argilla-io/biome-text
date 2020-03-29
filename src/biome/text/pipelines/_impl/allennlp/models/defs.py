import inspect
from abc import ABC
from typing import List, Type, Any

from allennlp.common import Params, Registrable
from allennlp.common.from_params import remove_optional
from allennlp.models import Model

from ...allennlp import helpers


class ChainedLayer(Registrable):
    """TBD"""

    @classmethod
    def from_params(cls: Type[Any], params: Params, **extras) -> Any:
        init_model_signature = inspect.signature(cls.__init__)

        args = {}
        prev_component = extras.get("prev_layer")  # This is a must
        for name, component_cfg in params.as_dict().items():
            parameter = init_model_signature.parameters.get(name)
            if not parameter:
                raise TypeError(f"'{name}' not found in init method for model '{cls}'")

            component_class = remove_optional(parameter.annotation)
            new_configuration = helpers.chain_component_config(
                component_class, component_cfg, prev_component
            )
            prev_component = helpers.component_from_config(
                component_class, new_configuration, previous=prev_component, **extras
            )
            args[name] = prev_component

        return cls(**args, **extras)


class ITextClassifier(Model, ChainedLayer, ABC):
    """Interface text classifier model"""

    def output_classes(self) -> List[str]:
        raise NotImplementedError

    def extend_labels(self, labels: List[str]):
        raise NotImplementedError
