import logging

from biome.allennlp.dataset_readers import SequenceClassifierDatasetReader
from biome.allennlp.dataset_readers.utils import get_reader_configuration
from biome.allennlp.models import SequencePairClassifier

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import logging
from inspect import signature
from typing import Dict, Iterable, Optional, Union, List

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from dask.dataframe import Series
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _ForwardConfiguration:
    def __init__(
        self,
        record1: Union[str, List[str]],
        record2: Union[str, List[str]],
        label: Union[str, dict] = None,
        target: dict = None,
    ):
        if isinstance(record1, str):
            record1 = [record1]
        if isinstance(record2, str):
            record2 = [record2]
        self._record1 = record1
        self._record2 = record2
        self._label = None
        self._default_label = None
        self._metadata = None

        if target and not label:
            label = target

        if label:
            if isinstance(label, str):
                self._label = label
            else:
                self._label = label.get("name", label.get("gold_label"))
                self._default_label = label.get(
                    "default", label.get("use_missing_label")
                )
                self._metadata = (
                    self._load_metadata(label.get("metadata_file"))
                    if label.get("metadata_file")
                    else None
                )

    @staticmethod
    def _load_metadata(path: str) -> Dict[str, str]:
        with open(path) as metadata_file:
            classes = [line.rstrip("\n").rstrip() for line in metadata_file]

        mapping = {idx + 1: cls for idx, cls in enumerate(classes)}
        # mapping variant with integer numbers
        mapping = {**mapping, **{str(key): value for key, value in mapping.items()}}

        return mapping

    @property
    def record1(self) -> List[str]:
        return self._record1

    @property
    def record2(self) -> List[str]:
        return self._record2

    @property
    def label(self) -> str:
        return self._label

    @property
    def default_label(self):
        return self._default_label

    @property
    def metadata(self):
        return self._metadata


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

        super(SequencePairClassifierDatasetReader, self).__init__()

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
        data_source, forward = get_reader_configuration(
            file_path, _ForwardConfiguration
        )

        ds_key = file_path
        dataset = self._cached_datasets.get(ds_key)
        if dataset is not None:
            logger.debug("Loaded cached dataset {}".format(file_path))
            return dataset
        else:
            logger.debug("Read dataset from {}".format(file_path))
            dataset = data_source.to_dataframe().compute()
            dataset["record1"] = (
                dataset[forward.record1]
                .astype(str)
                .apply(lambda r: " ".join(r.values), axis=1)
            )
            dataset["record2"] = (
                dataset[forward.record2]
                .astype(str)
                .apply(lambda r: " ".join(r.values), axis=1)
            )
            dataset["label"] = (
                dataset[forward.label]
                .astype(str)
                .apply(self.sanitize_label, forward=forward)
            )
            instances = (
                dataset[["record1", "record2", "label"]]
                .astype(str)
                .apply(self.example_to_instance, axis=1)
            )
            self._cached_datasets[ds_key] = instances
            return (instance for idx, instance in instances.iteritems() if instance)

    def example_to_instance(self, example: Union[dict, Series]) -> Optional[Instance]:
        """Extracts the forward parameters from the example and transforms them to an `Instance`

        Parameters
        ----------
        example
            The keys of this dictionary should match the arguments of the `forward` method of your model.

        Returns
        -------
        instance
            Returns `None` if the example could not be transformed to an Instance.
        """
        record1 = example["record1"]
        record2 = example["record2"]
        label = example.get("label", None)

        if not record1 or not record2:
            logger.warning(f"'record1 or record2' probably contains an empty string!")
            return None

        fields = {
            "record1": TextField(self.tokenizer.tokenize(record1), self.token_indexers),
            "record2": TextField(self.tokenizer.tokenize(record2), self.token_indexers),
        }
        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)
