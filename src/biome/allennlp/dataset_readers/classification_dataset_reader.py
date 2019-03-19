import logging
from inspect import signature, Parameter
from typing import Any, Dict, Iterable, Optional, Union

from allennlp.data import DatasetReader, Instance, TokenIndexer, Field, Tokenizer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from biome.allennlp.models import SequenceClassifier
from biome.data.sources import read_dataset, RESERVED_FIELD_PREFIX
from biome.data.utils import read_datasource_cfg
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sequence_classifier")
class SequenceClassifierDatasetReader(DatasetReader):
    """A DatasetReader for the SequenceClassifier model.

    Parameters
    ----------
    forward
        We want to get rid of this one, or at least infer it from the model type
    tokenizer
        By default we use a WordTokenizer with the SpacyWordSplitter
    token_indexers
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
        self.token_field_id = list(self.forward_params)[0]

        self.token_indexers = token_indexers or {self.tokens_field_id: SingleIdTokenIndexer}

        self._cached_datasets = dict()

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """

        Parameters
        ----------
        file_path
            Path to the configuration file (yml) of the data source.

        Yields
        ------

        """
        config = read_datasource_cfg(file_path)
        ds_key = id(config)

        dataset = self._cached_datasets.get(ds_key)
        if dataset:
            logger.debug("Loaded cached dataset {}".format(file_path))
        else:
            logger.debug("Read dataset from {}".format(file_path))
            dataset = read_dataset(config).persist()
            self._cached_datasets[ds_key] = dataset

        for example in dataset:
            instance = self._example_to_instance(example)
            if instance:
                yield instance

    def example_to_instance(self, example: Dict) -> Optional[Instance]:  # type: ignore
        """ Transforms an example to an Instance

        The actual work is done in the private method `_example_to_instance`.

        Parameters
        ----------
        example
            The keys of this dictionary should match the arguments of the `forward` method of your model.

        Returns
        -------
        instance
            Returns `None` if the example could not be transformed to an Instance.
        """
        # pylint: disable=arguments-differ
        logger.debug("Example:[%s]", example)
        try:
            if self.forward_definition:
                instance = self._instance_by_forward_definition(example)
            else:
                instance = self._instance_by_target_definition(example)  # deprecated
        except Exception as e:
            logger.warning(e)
            return None

        return instance

    def _example_to_instance(self, example: Dict) -> Optional[Instance]:
        """

        Parameters
        ----------
        example

        Returns
        -------

        """
        fields = {}
        try:
            for field, value in example.items():
                if not value:
                    raise ValueError(f"{field} contains an empty string!")
                fields[field] = self.value_to_field(field, value)
        except Exception as e:
            logger.warning(e)
            return None

        return Instance(fields)

    def value_to_field(self, field_type: str, value: Any) -> Union[LabelField, TextField]:
        """
        Parameters
        ----------
        field_type
        value

        Returns
        -------

        Raises
        ------
        """
        param = self.forward_params.get(field_type)
        if not param:
            raise ValueError(f"{field_type} must not form part of the Instance passed on to the model!")
        # the gold label should be optional in the model, otherwise no predict is possible
        if param.default is not Parameter.empty:
            return LabelField(value)
        else:
            return TextField(self.tokenizer.tokenize(value), self.token_indexers)



        for par_name, par in self.forward_params.items():
            if

        if field_type == self.gold_label_field_id:
            return LabelField(value)
        # allow multiple token fields to be passed on to the forward of the model
        elif field_type.startswith(self.tokens_field_id):  # allow multiple token fields to be passed on to the forward of the model
            return TextField(self.tokenizer.tokenize(value), self.token_indexers)
        else:
            raise ValueError(f"{field_type} must not form part of the Instance passed on to the model!")

    def _field_from_type(self, field_type: str, field_value: Any) -> Field:
        if field_type == "LabelField":
            return LabelField(field_value)
        elif field_type == "TextField":
            return TextField(self.tokenizer.tokenize(field_value), self.token_indexers)
        else:
            raise TypeError(f"{field_type} is not supported yet.")

    def _instance_by_forward_definition(self, example) -> Instance:
        def field_from_type(field_type: str, field_value: Any) -> Field:
            if field_type == "LabelField":
                return LabelField(field_value)
            elif field_type == "TextField":
                return TextField(self.tokenizer.tokenize(field_value), self.token_indexers)
            else:
                raise TypeError(f"{field_type} is not supported yet.")

        fields = {
            field: field_from_type(field_type, example[field])
            for field, field_type in self.forward_definition.items()
            if example.get(field) is not None
        }

        return Instance(fields)

    def _instance_by_target_definition(self, example) -> Instance:
        logger.warning(
            "Call to the deprecated method _instance_by_target_definition(). "
            "Use forward definition in config file instead of target."
        )
        fields: Dict[str, Field] = {}

        for field, value in example.items():
            if not self._is_reserved_field(field):
                tensor = (
                    LabelField(value)
                    if field == self.gold_label_id
                    else TextField(self.tokenizer.tokenize(value), self.token_indexers)
                )
                fields[field] = tensor

        return Instance(fields)

    def _is_reserved_field(self, field_name: str) -> bool:
        return field_name and field_name.startswith(RESERVED_FIELD_PREFIX)

"""
    @staticmethod
    def _text_to_instance(
        example: Dict,
        forward_definition: Dict[str, Any],
        gold_label_id: str,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer,
    ) -> Instance:
        def is_reserved_field(field_name: str) -> bool:
            return field_name and field_name.startswith(RESERVED_FIELD_PREFIX)

        def instance_by_forward_definition() -> Instance:
            def field_from_type(field_type: str, field_value: Any) -> Field:
                if field_type == "LabelField":
                    return LabelField(field_value)
                elif field_type == "TextField":
                    return TextField(tokenizer.tokenize(field_value), token_indexers)
                else:
                    raise TypeError(f"{field_type} is not supported yet.")

            fields = {
                field: field_from_type(field_type, example[field])
                for field, field_type in forward_definition.items()
                if example.get(field) is not None
            }

            return Instance(fields)

        def instance_by_target_definition() -> Instance:
            logger.warning(
                "Call to the deprecated method instance_by_target_definition(). "
                "Use forward definition in config file instead of target."
            )
            fields: Dict[str, Field] = {}

            for field, value in example.items():
                if not is_reserved_field(field):
                    tensor = (
                        LabelField(value)
                        if field == gold_label_id
                        else TextField(tokenizer.tokenize(value), token_indexers)
                    )
                    fields[field] = tensor

            return Instance(fields)

        return (
            instance_by_forward_definition()
            if forward_definition
            else instance_by_target_definition()
        )
"""

@DatasetReader.register("sequence_pair_classifier")
class SequencePairClassifierDatasetReader(SequenceClassifierDatasetReader):
    """

    """
    def __init__(self):
        super(SequenceClassifierDatasetReader, self).__init__(lazy=True)


