import logging
from inspect import signature, Parameter
from typing import Dict, Iterable, Optional, Union, Any

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from biome.allennlp.models import SequenceClassifier
from biome.data.sources import DataSource
from .mixins import TextFieldBuilderMixin, CacheableMixin

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sequence_classifier")
class SequenceClassifierDatasetReader(
    DatasetReader, TextFieldBuilderMixin, CacheableMixin
):
    """A DatasetReader for the SequenceClassifier model.

    Parameters
    ----------
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter

    token_indexers
        By default we use the following dict {'tokens': SingleIdTokenIndexer}

    as_text_field
        Build ``Instance`` fields as ``ListField`` of ``TextField`` or ``TextField``
    """

    # Name of the "label parameter" in the model's forward method
    LABEL_TAG = "label"

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
            token_indexers=token_indexers or {"tokens": SingleIdTokenIndexer()},
            as_text_field=as_text_field,
        )

        # The keys of the Instances have to match the signature of the forward method of the model
        self.forward_params = self.get_forward_signature(SequenceClassifier)

    @staticmethod
    def get_forward_signature(model: "allennlp.models.Model") -> Dict[str, "Parameter"]:
        """Get the parameter names and `Parameter`s of the model's forward method.

        Returns
        -------
        parameters
            A dict mapping the parameter name to a `Parameter` instance.
        """
        parameters = dict(signature(model.forward).parameters)
        # the forward method is always a non-static method of the model class -> remove self
        del parameters["self"]

        return parameters

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
            dataset = data_source.to_forward_dataframe()
            instances = dataset.apply(self.text_to_instance, axis=1, meta=(None, "object"))

            # cache instances of the data set
            self.set(file_path, instances)

        return (instance for idx, instance in instances.iteritems() if instance)

    def text_to_instance(
        self, text: Union[Dict, "dask.Series", "pandas.Series"]
    ) -> Optional[Instance]:
        """Extracts the forward parameters from the example and transforms them to an `Instance`

        Parameters
        ----------
        text
            The keys of this dict should contain the parameter names of the `forward` method of your model.

        Returns
        -------
        instance
            Returns `None` if the example could not be transformed to an Instance.
        """
        fields = {}
        for param_name, param in self.forward_params.items():
            try:
                value = text[param_name]
                fields[param_name] = self._value_to_field(param_name, value)

            except KeyError as e:
                # if parameter is required by the forward method raise a meaningful error
                if param.default is Parameter.empty:
                    raise KeyError(
                        f"{e}; You are probably missing '{param_name}' in your forward definition of the data source."
                    )

        return Instance(fields)

    def _value_to_field(
        self, param_name: str, value: Any
    ) -> Union[LabelField, TextField, ListField]:
        """Embeds the value in one of the `allennlp.data.fields`.
        For now the field type is basically inferred from the hardcoded label tag ...

        Parameters
        ----------
        param_name
            Must match one of the parameters in the `forward` method of your model.
        value
            Value of the field.

        Returns
        -------
        Returns either a `LabelField` or a `TextField`/`ListField` depending on the `param_name`.
        """
        if param_name == self.LABEL_TAG:
            return LabelField(value)
        else:
            return self.build_textfield(value)
