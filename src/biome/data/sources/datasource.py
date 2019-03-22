import logging
from typing import Dict, Optional, TypeVar, Type

import yaml
from dask.bag import Bag

from biome.data.sources.readers import from_csv, from_json, from_excel, from_elasticsearch
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
    kwargs
        Additional kwargs are passed on to the *source readers* that depend on the format.
    """

    # maps the supported formats to the corresponding "source readers"
    SUPPORTED_FORMATS = {
        "xls": (from_excel, dict(na_filter=False, keep_default_na=False, dtype=str)),
        "xlsx": (from_excel, dict(na_filter=False, keep_default_na=False, dtype=str)),
        "csv": (from_csv, dict(assume_missing=False, na_filter=False, dtype=str)),
        "json": (from_json, dict()),
        "jsonl": (from_json, dict()),
        "json-l": (from_json, dict()),
        "elasticsearch": (from_elasticsearch, dict()),
        "elastic": (from_elasticsearch, dict()),
        "es": (from_elasticsearch, dict()),
    }

    def __init__(
        self, format: str, forward: Dict, **kwargs
    ):
        self.format = format
        self.forward = forward.copy()
        self.kwargs = kwargs

        self.example_preparator = ExamplePreparator(forward)

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
        data_source = self._build_data_source()
        examples = data_source.map(self._extract_example, include_source)

        return examples.filter(lambda example: example is not None)

    def _build_data_source(self) -> Bag:
        """Reads in the data source and returns it as dicts in a `dask.Bag`

        Returns
        -------
        bag
            A `dask.Bag` of dicts.
        """
        try:
            from_source_to_bag, arguments = self.SUPPORTED_FORMATS[self.format]
            return from_source_to_bag(**{**arguments, **self.kwargs})
        except KeyError:
            raise TypeError(
                "Format {} not supported. Supported formats are: {}".format(
                    self.format, " ".join(self.SUPPORTED_FORMATS)
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
        return cls(**cfg_dict)
