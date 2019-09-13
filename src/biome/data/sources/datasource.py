import logging
import os.path
from typing import Dict, TypeVar, Type, Callable, Any, Union

import pandas as pd
import yaml
from dask.bag import Bag
from dask.dataframe import DataFrame

from biome.data.sources.readers import (
    from_csv,
    from_json,
    from_excel,
    from_elasticsearch,
    from_parquet,
)
from biome.data.sources.utils import make_paths_relative

# https://stackoverflow.com/questions/51647747/how-to-annotate-that-a-classmethod-returns-an-instance-of-that-class-python
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
        except KeyError:
            raise TypeError(
                f"Format {format} not supported. Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        df = source_reader(**{**arguments, **kwargs}).dropna(how="all")
        df = df.rename(
            columns={column: column.strip() for column in df.columns.astype(str).values}
        )
        if "id" in df.columns:
            df = df.set_index("id")

        self._forward = ClassificationForwardConfiguration(**forward)
        self._df = df

    @property
    def forward(self) -> dict:
        return self._forward

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

    def to_forward_dataframe(self) -> pd.DataFrame:
        """
        Adds columns to the DataFrame that are named after the parameter names in the model's forward method.
        The content of these columns is specified in the forward dictionary of the data source yaml file.

        Returns
        -------
        forward_dataframe
            Contains additional columns corresponding to the parameter names of the model's forward method.
        """
        forward_dataframe = self._df.compute()

        forward_dataframe["label"] = (
            forward_dataframe[self._forward.label]
            .astype(str)
            .apply(self._forward.sanitize_label)
        )
        for forward_token_name, data_column_names in self._forward.tokens.items():
            # convert str to list, otherwise the axis=1 raises an error with the returned pd.Series
            data_column_names = (
                [data_column_names]
                if isinstance(data_column_names, str)
                else data_column_names
            )
            forward_dataframe[forward_token_name] = forward_dataframe[
                data_column_names
            ].apply(lambda x: x.to_dict(), axis=1)
            # if the data source df already has a column with the forward_token_name, it will be replaced!

        return forward_dataframe

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

        # File system paths are usually specified relative to the yaml config file -> they have to be modified
        # path_keys is not necessary, but specifying the dict keys
        # (for which we check for relative paths) is a safer choice
        path_keys = ["path", "metadata_file"]
        make_paths_relative(os.path.dirname(file_path), cfg_dict, path_keys=path_keys)

        return cls(**cfg_dict)


class ClassificationForwardConfiguration(object):
    """
        This ``ClassificationForwardConfiguration`` represents  forward operations for label
        configuration in classification problems.

        Parameters
        ----------

        label:
            Name of the label column in the data
        target:
            (deprecated) Just an alias for label
        tokens:
            These kwargs match the token names (of the model's forward method)
            to the column names (of the data).
    """

    def __init__(self, label: Union[str, dict] = None, target: dict = None, **tokens):
        self._label = None
        self._default_label = None
        self._metadata = None

        if target and not label:
            label = target

        if label:
            if isinstance(label, str):
                self._label = label
            else:
                self._label = (
                    label.get("name")
                    or label.get("label")
                    or label.get("gold_label")
                    or label.get("field")
                )
                if not self._label:
                    raise RuntimeError("I am missing the label name!")
                self._default_label = label.get(
                    "default", label.get("use_missing_label")
                )
                self._metadata = (
                    self.load_metadata(label.get("metadata_file"))
                    if label.get("metadata_file")
                    else None
                )

        self.tokens = tokens

    @staticmethod
    def load_metadata(path: str) -> Dict[str, str]:
        with open(path) as metadata_file:
            classes = [line.rstrip("\n").rstrip() for line in metadata_file]

        mapping = {idx + 1: cls for idx, cls in enumerate(classes)}
        # mapping variant with integer numbers
        mapping = {**mapping, **{str(key): value for key, value in mapping.items()}}

        return mapping

    @property
    def label(self) -> str:
        return self._label

    @property
    def default_label(self):
        return self._default_label

    @property
    def metadata(self):
        return self._metadata

    def sanitize_label(self, label: str) -> str:
        label = label.strip()
        if self.default_label:
            label = label if label else self.default_label
        if self.metadata:
            label = self.metadata.get(label, label)

        return label
