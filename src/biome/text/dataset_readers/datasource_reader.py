import inspect
import logging
from inspect import Parameter
from typing import Iterable, Dict

from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer

from biome.data.sources import DataSource
from biome.text.dataset_readers.mixins import TextFieldBuilderMixin, CacheableMixin


class DataSourceReader(DatasetReader, TextFieldBuilderMixin, CacheableMixin):
    """
    A DataSetReader as base for read instances from ``DataSource``

    The subclasses must implements their own way to transform input data to ``Instance``
    in the text_to_field method

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
        skip_empty_tokens: bool = False,
    ) -> None:
        DatasetReader.__init__(self, lazy=True)
        TextFieldBuilderMixin.__init__(
            self,
            tokenizer=tokenizer,
            # The token_indexers keys are directly related to the model text_field_embedder configuration
            token_indexers=token_indexers,
            as_text_field=as_text_field,
        )
        self._skip_empty_tokens = skip_empty_tokens

        self._signature = {
            name: dict(optional=value.default != Parameter.empty)
            for name, value in inspect.signature(
                self.text_to_instance
            ).parameters.items()
            if name != "self"
        }

    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

    @property
    def signature(self) -> dict:
        """
        Describe de input signature for the pipeline predictions

        Returns
        -------
            A list of expected inputs with information about if input is optional or nor.

            For example, for the signature
            >>def text_to_instance(a:str,b:str, c:str=None)

            This method will return:
            >>{"a":{"optional": False},"b":{"optional": False},"c":{"optional": True}}
        """
        return self._signature.copy()

    def _read(self, file_path: str) -> Iterable[Instance]:
        """An generator that yields `Instance`s that are fed to the model

        This method is implicitly called when training the model.
        The predictor uses the `self.text_to_instance_with_data_filter` method.

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
            self.logger.debug("Loaded cached data set {}".format(file_path))
        else:
            self.logger.debug("Read data set from {}".format(file_path))
            dataset = data_source.to_mapped_dataframe()
            instances = dataset.apply(
                lambda x: self.text_to_instance(**x.to_dict()),
                axis=1,
                meta=(None, "object"),
            )

            # cache instances of the data set
            self.set(file_path, instances)

        return (instance for idx, instance in instances.iteritems() if instance)

    def text_to_instance(self, **inputs) -> Instance:
        raise NotImplementedError
