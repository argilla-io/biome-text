import logging
from typing import Dict, Iterable, Optional, Union, List

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.fields import LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from biome.allennlp.dataset_readers.classification_forward_configuration import (
    ClassificationForwardConfiguration,
)
from biome.allennlp.dataset_readers.mixins import TextFieldBuilderMixin, CacheableMixin
from biome.allennlp.dataset_readers.utils import get_reader_configuration
from biome.data.sources import DataSource
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SequencePairClassifierForwardConfiguration(ClassificationForwardConfiguration):
    def __init__(
        self,
        record1: Union[str, List[str]],
        record2: Union[str, List[str]],
        label: Union[str, dict] = None,
        target: dict = None,
    ):
        super(SequencePairClassifierForwardConfiguration, self).__init__(label, target)
        self._record1 = [record1] if isinstance(record1, str) else record1
        self._record2 = [record2] if isinstance(record2, str) else record2

    @property
    def record1(self) -> List[str]:
        return self._record1

    @property
    def record2(self) -> List[str]:
        return self._record2


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
            token_indexers=(
                token_indexers
                or {
                    "record1": SingleIdTokenIndexer(),
                    "record2": SingleIdTokenIndexer(),
                }
            ),
            as_text_field=as_text_field,
        )

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
            file_path, SequencePairClassifierForwardConfiguration
        )

        ds_key = file_path
        dataset = self.get(ds_key)
        if dataset is not None:
            logger.debug("Loaded cached dataset {}".format(file_path))
            return dataset
        else:
            logger.debug("Read dataset from {}".format(file_path))
            dataset = self._read_as_forward_dataset(data_source, forward)
            instances = dataset[["record1", "record2", "label"]].apply(
                self.text_to_instance, axis=1
            )
            self.set(ds_key, instances)
            return (instance for idx, instance in instances.iteritems() if instance)

    @staticmethod
    def _read_as_forward_dataset(
        data_source: DataSource, forward: SequencePairClassifierForwardConfiguration
    ):
        dataset = data_source.to_dataframe().compute()
        dataset["record1"] = dataset[forward.record1].apply(
            lambda x: x.to_dict(), axis=1
        )
        dataset["record2"] = dataset[forward.record2].apply(
            lambda x: x.to_dict(), axis=1
        )
        dataset["label"] = (
            dataset[forward.label].astype(str).apply(forward.sanitize_label)
        )
        return dataset

    def text_to_instance(
        self, example: Dict[str, Union[str, Dict[str, str]]]
    ) -> Optional[Instance]:
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

        # `record1` and `record2` must be a dictionary with original
        # data values defined in forward configuration
        record1_field = self.build_textfield(example["record1"])
        record2_field = self.build_textfield(example["record2"])

        if not record1_field or not record2_field:
            logger.warning(f"'record1 or record2' probably contains no info")
            return None

        fields = {"record1": record1_field, "record2": record2_field}

        label = example.get("label", None)
        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)
