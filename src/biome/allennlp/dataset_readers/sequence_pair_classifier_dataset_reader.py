import logging
from typing import Dict, Iterable, Optional, Union, Any
from inspect import signature, Parameter

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from biome.data.sources import DataSource
from biome.allennlp.models import SequencePairClassifier
from .mixins import TextFieldBuilderMixin, CacheableMixin

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sequence_pair_classifier")
class SequencePairClassifierDatasetReader(
    DatasetReader, TextFieldBuilderMixin, CacheableMixin
):
    """A DatasetReader for the SequencePairClassifier model.

    Parameters
    ----------
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter
    token_indexers
        By default we use a SingleIdTokenIndexer for all token fields
    as_text_field
        False by default, if enabled, the ``Instances`` generated will contains
        the input text fields as a concatenation of input fields in a single ``TextField``.
        By default, the reader will use a ``ListField`` of ``TextField`` for input representation
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        as_text_field: bool = False,
    ) -> None:
        DatasetReader.__init__(self, lazy=True)
        TextFieldBuilderMixin.__init__(
            self,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            as_text_field=as_text_field,
        )

        # The keys of the Instances have to match the signature of the forward method of the model
        self.forward_params = signature(SequencePairClassifier.forward).parameters

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """An generator that yields `Instance`s that are fed to the model

        This method is implicitly called when training the model.
        The predictor uses the `self.text_to_instance` method.

        Parameters
        ----------
        file_path
            Path to the configuration file (yml) of the data source.

        Yields
        ------
        instance
            An `Instance` that is fed to the model
        """
        data_source = DataSource.from_yaml(file_path)

        # get cached instances of the data set
        instances = self.get(file_path)
        if instances is not None:
            logger.debug("Loaded cached data set {}".format(file_path))
        else:
            logger.debug("Read data set from {}".format(file_path))
            dataset = data_source.read_as_forward_dataset()
            instances = dataset.apply(self.text_to_instance, axis=1)

            # cache instances of the data set
            self.set(file_path, instances)

            # If memory is an issue maybe we should only cache the dataset and yield instances:
            # for example in dataset.itertuples(index=False):
            #     yield self.text_to_instance(example)

        return (instance for idx, instance in instances.iteritems() if instance)

    def text_to_instance(
            self, example: Dict[str, str], exclude_optional: bool = False
    ) -> Optional[Instance]:
        """Extracts the forward parameters from the example and transforms them to an `Instance`

        Parameters
        ----------
        example
            The keys of this dictionary should match the arguments of the `forward` method of your model.
        exclude_optional
            Only extract the mandatory parameters of the model's forward method.

        Returns
        -------
        instance
            Returns `None` if the example could not be transformed to an Instance.
        """
        fields = {}
        try:
            for param_name, param in self.forward_params.items():
                if param_name == "self":
                    continue
                # if desired, skip optional parameters, like the label for example
                if exclude_optional and param.default is not Parameter.empty:
                    continue

                value = getattr(example, param_name)
                if not value:
                    raise ValueError(f"{param_name} probably contains an empty string!")
                fields[param_name] = self._value_to_field(param_name, value)
        except ValueError as e:
            logger.warning(e)
            return None

        return Instance(fields)

    def _value_to_field(
            self, field_type: str, value: Any
    ) -> Union[LabelField, TextField]:
        """Embeds the value in one of the `allennlp.data.fields`

        Parameters
        ----------
        field_type
            Name of the field, must match one of the parameters in the `forward` method of your model.
        value
            Value of the field.

        Returns
        -------
        Returns either a `LabelField` or a `TextField` depending on the `field_type` parameter.
        """
        param = self.forward_params.get(field_type)
        # the label must be optional in a classification model, otherwise no predict is possible
        if param.default is not Parameter.empty:
            return LabelField(value)
        else:
            return self.build_textfield(value)
