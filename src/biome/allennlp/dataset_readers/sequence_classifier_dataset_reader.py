import logging
from typing import Dict, Iterable, Optional, Union, List

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.fields import LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from biome.allennlp.dataset_readers.classification_forward_configuration import (
    ClassificationForwardConfiguration,
)
from biome.allennlp.dataset_readers.mixins import TextFieldBuilderMixin, CacheableMixin
from biome.allennlp.dataset_readers.utils import get_reader_configuration
from biome.data.sources import DataSource

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SequenceClassifierForwardConfiguration(ClassificationForwardConfiguration):
    def __init__(
        self,
        tokens: Union[str, List[str]],
        label: Union[str, dict] = None,
        target: dict = None,
    ):
        super(SequenceClassifierForwardConfiguration, self).__init__(label, target)
        self._tokens = [tokens] if isinstance(tokens, str) else tokens

    @property
    def tokens(self) -> List[str]:
        return self._tokens.copy()


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
            token_indexers=token_indexers or {"tokens": SingleIdTokenIndexer()},
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
            file_path, SequenceClassifierForwardConfiguration
        )

        ds_key = file_path
        dataset = self.get(ds_key)
        if dataset is not None:
            logger.debug(f"Loaded cached dataset {file_path}")
            return dataset
        else:
            logger.debug(f"Read dataset from {file_path}")
            dataset = self._read_as_forward_dataset(data_source, forward)
            instances = dataset[["tokens", "label"]].apply(
                self.text_to_instance, axis=1
            )
            self.set(ds_key, instances)
            return (instance for idx, instance in instances.iteritems() if instance)

    @staticmethod
    def _read_as_forward_dataset(
        data_source: DataSource, forward: SequenceClassifierForwardConfiguration
    ):
        dataset = data_source.to_dataframe().compute()

        dataset["tokens"] = dataset[forward.tokens].apply(lambda x: x.to_dict(), axis=1)
        dataset["label"] = (
            dataset[forward.label].astype(str).apply(forward.sanitize_label)
        )

        return dataset

    @overrides
    def text_to_instance(
        self, example: Dict[str, Union[Dict[str, str], str]]
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

        tokens_field = self.build_textfield(example["tokens"])
        if not tokens_field:
            logger.warning(f"'tokens' probably contains an empty string!")
            return None

        fields = {"tokens": tokens_field}
        label = example.get("label", None)
        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)
