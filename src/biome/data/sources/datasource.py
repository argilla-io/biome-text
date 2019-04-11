import logging
from typing import Dict, Optional, TypeVar, Type, Callable, Any

import yaml
from dask.bag import Bag

from biome.data.sources.example_preparator import ExamplePreparator

# https://stackoverflow.com/questions/51647747/how-to-annotate-that-a-classmethod-returns-an-instance-of-that-class-python
T = TypeVar("T")

_logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataSource:
    """This class takes care of reading the data source, usually specified in a yaml file.

    It uses the *source readers* to extract a dask Bag of dicts,
    From that it extracts examples via the `ExamplePreparator`.

    Parameters
    ----------
    format
        The data format. Supported formats are listed as keys in the `SUPPORTED_FORMATS` dict of this class.
    forward
        A dict passed on to the `ExamplePreparator`.
        The keys of this dict must match the arguments of the model's `forward` method.
    definition_file
        Path to data source yaml specification.
        This path could be useful for file-based readers
    kwargs
        Additional kwargs are passed on to the *source readers* that depend on the format.
    """

    # maps the supported formats to the corresponding "source readers"
    SUPPORTED_FORMATS = dict()

    def __init__(
        self, format: str, forward: Dict = None, definition_file: str = None, **kwargs
    ):
        self.format = format
        self.example_preparator = ExamplePreparator(forward.copy()) if forward else None
        self.kwargs = kwargs
        self.definition_file = definition_file

    @classmethod
    def add_supported_format(
        cls, format_key: str, parser: Callable, default_params: Dict[str, Any] = None
    ) -> None:
        """Add a new format and reader to the data source readers.

        Parameters
        ----------
        format_key
            The new format key
        parser
            The parser function
        default_params
            Default parameters for the parser function
        """
        if format_key in cls.SUPPORTED_FORMATS.keys():
            _logger.warning("Already defined format {}".format(format_key))
            return

        cls.SUPPORTED_FORMATS[format_key] = (parser, default_params)

    def read(self, include_source: bool = False) -> Bag:
        """Reads a data source and extracts the relevant information.

        Parameters
        ----------
        include_source
            If True, the returned dicts include a *source* key that holds the entire source dict.

        Returns
        -------
        bag
            A `dask.Bag` of dicts (called examples) that hold the relevant information passed on to our model
            (for example the tokens and the label).
        """
        if not ExamplePreparator:
            raise RuntimeError(
                "You properly forgot to specify the forward configuration."
            )
        data_source = self.to_bag()
        examples = data_source.map(self._extract_example, include_source)

        return examples.filter(lambda example: example is not None)

    def to_bag(self) -> Bag:
        """Reads in the data source and returns it as dicts in a `dask.Bag`

        Returns
        -------
        bag
            A `dask.Bag` of dicts.
        """
        try:
            from_source_to_bag, arguments = self.SUPPORTED_FORMATS[self.format]
            if isinstance(from_source_to_bag, DataSourceReader):
                return from_source_to_bag(self)
            return from_source_to_bag(**{**arguments, **self.kwargs})
        except KeyError:
            raise TypeError(
                "Format [{}] not supported. Supported formats are: {}".format(
                    self.format, ", ".join(self.SUPPORTED_FORMATS)
                )
            )

    def _extract_example(
        self, source_dict: Dict, include_source: bool
    ) -> Optional[Dict]:
        """Extracts the relevant information from a source dict.

        Parameters
        ----------
        source_dict
            A single entry of the data source as a dictionary.
        include_source
            If True, the returned dict includes a *source* key that holds the entire source dict.

        Returns
        -------
        example
            A dict with keys that match the arguments of our model's forward method.
            If the extraction fails, None is returned.
        """
        try:
            return self.example_preparator.read_info(source_dict, include_source)
        except Exception as e:
            _logger.warning(e)
            return None

    @classmethod
    def from_yaml(cls: Type[T], file_path: str) -> T:
        """Create a data source from a yaml file.

        The yaml file has to serialize a dict, whose keys matches the arguments of the `DataSource` class.

        Parameters
        ----------
        file_path
            The path to the yaml file.

        Returns
        -------
        cls
        """
        with open(file_path) as yaml_file:
            cfg_dict = yaml.safe_load(yaml_file)
        return cls(**cfg_dict, definition_file=file_path)


class DataSourceReader(object):
    """
    Abstract class for data source readers definitions

    This class extends data source readers functionality by passing the whole data source instance
    """

    def __call__(self, data_source: DataSource):
        raise NotImplementedError
