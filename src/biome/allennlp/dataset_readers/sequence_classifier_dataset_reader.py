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
    def get_forward_signature(model: "Model") -> Dict[str, "Parameter"]:
        """Get the parameter names and `Parameter`s of the model's forward method.

        Returns
        -------
        parameters
            A dict mapping the parameter name to a `Parameter` instance.
        """
        parameters = dict(signature(model.forward).parameters)
        # the forward method is always a non-static method of the model class -> remove self
        del parameters["self"]
        del parameters[
            SequenceClassifierDatasetReader.LABEL_TAG
        ]  # `label` is the mandatory name for this kind of models

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
            instances = dataset.apply(self.text_to_instance, axis=1)

            # cache instances of the data set
            self.set(file_path, instances)

            # If memory is an issue maybe we should only cache the dataset and yield instances:
            # for example in dataset.itertuples(index=False):  # itertuples returns `collections.namedtuple`s !!!
            #     yield self.text_to_instance(example)

        return (instance for idx, instance in instances.iteritems() if instance)

    def text_to_instance(
        self, example: Union["pandas.Series", "collections.namedtuple"]
    ) -> Optional[Instance]:
        """Extracts the forward parameters from the example and transforms them to an `Instance`

        Parameters
        ----------
        example
            The keys of this `pandas.Series` should match the arguments of the `forward` method of your model.

        Returns
        -------
        instance
            Returns `None` if the example could not be transformed to an Instance.
        """
        fields = {}
        for param_name, param in self.forward_params.items():
            try:
                # getattr works for `pandas.Series` and `collections.namedtuple` !
                value = getattr(example, param_name)
                fields[param_name] = self.build_textfield(value)
            except AttributeError as e:
                # if parameter is required by the forward method raise a meaningful error
                if param.default is Parameter.empty:
                    raise AttributeError(
                        f"{e}; You are probably missing '{param_name}' in your forward definition of the data source."
                    )

        label_value = getattr(example, SequenceClassifierDatasetReader.LABEL_TAG)
        if label_value:
            fields[SequenceClassifierDatasetReader.LABEL_TAG] = LabelField(label_value)

        return Instance(fields)
