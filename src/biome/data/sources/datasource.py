import logging
import os.path
from typing import Dict, TypeVar, Type, Callable, Any

import yaml
from biome.data.sources.readers import (
    from_csv,
    from_json,
    from_excel,
    from_elasticsearch,
    from_parquet,
)
from biome.data.sources.utils import make_paths_relative
from dask.bag import Bag

# https://stackoverflow.com/questions/51647747/how-to-annotate-that-a-classmethod-returns-an-instance-of-that-class-python
from dask.dataframe import DataFrame

T = TypeVar("T")

_logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from .utils import row2dict


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
        "parquet": (from_parquet, dict()),
    }

    def __init__(self, format: str, forward: Dict = None, **kwargs):
        try:
            clean_format = format.lower().strip()
            source_reader, arguments = self.SUPPORTED_FORMATS[clean_format]
            df = source_reader(**{**arguments, **kwargs}).dropna(how="all")
            df = df.rename(
                columns={
                    column: column.strip() for column in df.columns.astype(str).values
                }
            )
            if "id" in df.columns:
                df = df.set_index("id")

            self._df = df
        except KeyError:
            raise TypeError(
                f"Format {format} not supported. Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
            )

    @classmethod
    def add_supported_format(
        cls, format_key: str, parser: Callable, default_params: Dict[str, Any]
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
        return self.to_bag()

    def to_dataframe(self) -> DataFrame:
        return self._df

    def to_bag(self) -> Bag:
        """Reads in the data source and returns it as dicts in a `dask.Bag`

        Returns
        -------
        bag
            A `dask.Bag` of dicts.
        """
        return self._df.to_bag(index=True).map(
            row2dict, columns=[str(column).strip() for column in self._df.columns]
        )

    @classmethod
    def from_cfg(cls: Type[T], cfg: dict, cfg_file: str) -> T:
        """ Create a data source from a dictionary configuration"""
        # File system paths are usually specified relative to the yaml config file -> they have to be modified
        path_keys = [
            "path",
            "metadata_file",
        ]  # specifying the dict keys is a safer choice ...
        make_paths_relative(os.path.dirname(cfg_file), cfg, path_keys=path_keys)
        return cls(**cfg)

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

        return DataSource.from_cfg(cfg_dict, file_path)
