import logging
from inspect import signature, Parameter
from typing import Any, Dict, Iterable, Optional, Union

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

from biome.allennlp.models import SequenceClassifier, SequencePairClassifier
from biome.data.sources import DataSource

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sequence_classifier")
class SequenceClassifierDatasetReader(DatasetReader):
    """A DatasetReader for the SequenceClassifier model.

    Parameters
    ----------
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter
    token_indexers
        By default we use a SingleIdTokenIndexer for all token fields
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:

        super().__init__(lazy=True)

        self.tokenizer = tokenizer or WordTokenizer()

        # The keys of the Instances have to match the signature of the forward method of the model
        self.forward_params = signature(SequenceClassifier.forward).parameters
        self.tokens_field_id = list(self.forward_params)[0]

        self.token_indexers = token_indexers or {
            self.tokens_field_id: SingleIdTokenIndexer
        }

        self._cached_datasets = dict()

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """An generator that yields `Instance`s that are fed to the model

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
        ds_key = id(data_source)

        dataset = self._cached_datasets.get(ds_key)
        if dataset:
            logger.debug("Loaded cached dataset {}".format(file_path))
        else:
            logger.debug("Read dataset from {}".format(file_path))
            dataset = data_source.read().persist()
            self._cached_datasets[ds_key] = dataset

        for example in dataset:
            instance = self.text_to_instance(example)
            if instance:
                yield instance

    @overrides
    def text_to_instance(self, example: Dict[str, str]) -> Optional[Instance]:
        """Transforms an example to an Instance

        Parameters
        ----------
        example
            The keys of this dictionary should match the arguments of the `forward` method of your model.

        Returns
        -------
        instance
            Returns `None` if the example could not be transformed to an Instance.

        Raises
        ------
        ValueError
            If a value in the `example` dict resolves to False.
        """
        fields = {}
        try:
            for field, value in example.items():
                if not value:
                    raise ValueError(f"{field} probably contains an empty string!")
                fields[field] = self._value_to_field(field, value)
        except Exception as e:
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
            Name of the field, must match one of the arguments in the `forward` method of your model.
        value
            Value of the field.

        Returns
        -------
        Returns either a `LabelField` or a `TextField` depending on the `field_type` parameter.

        Raises
        ------
        ValueError
            If `field_type` is not found in the arguments of your model's `forward` method.
        """
        param = self.forward_params.get(field_type)
        if not param:
            raise ValueError(
                f"{field_type} must not form part of the Instance passed on to the model!"
            )
        # the gold label must be optional in the classification model, otherwise no predict is possible
        if param.default is not Parameter.empty:
            return LabelField(value)
        else:
            return TextField(self.tokenizer.tokenize(value), self.token_indexers)


@DatasetReader.register("sequence_pair_classifier")
class SequencePairClassifierDatasetReader(SequenceClassifierDatasetReader):
    """A DatasetReader for the SequencePairClassifier model.

    Parameters
    ----------
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter
    token_indexers
        By default we use a SingleIdTokenIndexer for all token fields
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:

        super(SequenceClassifier, self).__init__(lazy=True)

        self.tokenizer = tokenizer or WordTokenizer()

        # The keys of the Instances have to match the signature of the forward method of the model
        self.forward_params = signature(SequencePairClassifier.forward).parameters
        self.tokens_field_id = list(self.forward_params)[0]
        self.tokens2_field_id = list(self.forward_params)[1]

        self.token_indexers = token_indexers or {
            self.tokens_field_id: SingleIdTokenIndexer,
            self.tokens2_field_id: SingleIdTokenIndexer,
        }

        self._cached_datasets = dict()
