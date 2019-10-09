import logging
from typing import Dict, Iterable, Optional, Union, Any, List

from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from biome.data.sources import DataSource
from overrides import overrides

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
            # TODO check if we can change arbritary the "tokens" key name without affecting the pipeline
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
        data_source = DataSource.from_yaml(file_path)

        # get cached instances of the data set
        instances = self.get(file_path)
        if instances is not None:
            logger.debug("Loaded cached data set {}".format(file_path))
        else:
            logger.debug("Read data set from {}".format(file_path))
            dataset = data_source.to_forward_dataframe()
            instances = dataset.apply(
                lambda d: self.text_to_instance(
                    **{str(k): v for k, v in d.to_dict().items()}
                ),
                axis=1,
                meta=(None, "object")
            ).dropna()

            # cache instances of the data set
            self.set(file_path, instances)

        return (instance for idx, instance in instances.iteritems() if instance)

    def text_to_instance(
        self,
        tokens: Union[str, List[str], dict],
        label: Optional[str] = None,
        # TODO we can use the extra args for metadata build. Not used for now but required
        **extra_args
    ) -> Optional[Instance]:
        """Extracts the forward parameters from the example and transforms them to an `Instance`

        Parameters
        ----------
        tokens
            The input tokens key,values (or the text string)
        label
            The label value

        Returns
        -------
        instance
            Returns `None` if cannot generate an new Instance.
        """
        fields = {}

        tokens_field = self.build_textfield(tokens)
        label_field = LabelField(label) if label else None

        if tokens_field:
            fields["tokens"] = tokens_field
        if label_field:
            fields["label"] = label_field

        return Instance(fields) if fields and len(fields) > 0 else None


@DatasetReader.register("bert_for_classification")
class BertForClassificationDatasetReader(SequenceClassifierDatasetReader):
    """A DatasetReader for the BertForClassification model.

    Since the forward signature is the same for our SequenceClassifier and the BertForClassification,
    we can just use the DatasetReader for the former model.
    We just register it with the same name as the model, to be consistent with our approach:
    Each model needs it own DatasetReader -> no need to specify the DatasetReader type in the model.yml
    """

    pass
