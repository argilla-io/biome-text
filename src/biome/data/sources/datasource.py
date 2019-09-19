import logging
import os.path
from typing import Dict, Callable, Any, Union

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

_logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from .utils import row2dict


class DataSource:
    """This class takes care of reading the data source, usually specified in a yaml file.

    It uses the *source readers* to extract a `dask.DataFrame`.

    Parameters
    ----------
    format
        The data format. Supported formats are listed as keys in the `SUPPORTED_FORMATS` dict of this class.
    forward
        An instance of a `ClassificationForwardConfiguration`
        Used to pass on the right parameters to the model's forward method.
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

    def __init__(
        self,
        format: str,
        forward: "ClassificationForwardConfiguration" = None,
        **kwargs,
    ):
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
        self._df = df

        self.forward = forward

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

    def to_bag(self) -> Bag:
        """Turns the DataFrame of the data source into a `dask.Bag` of dictionaries, one dict for each row.
        Each dictionary has the column names as keys.

        Returns
        -------
        bag
            A `dask.Bag` of dicts.
        """
        dict_keys = [str(column).strip() for column in self._df.columns]

        return self._df.to_bag(index=True).map(row2dict, columns=dict_keys)

    def to_forward_bag(self) -> Bag:
        """Turns the forward DataFrame of the data source into a `dask.Bag` of dictionaries, one dict for each row.
        Each dictionary has the column names as keys.

        Returns
        -------
        bag
            A `dask.Bag` of dicts.
        """
        forward_df = self.to_forward_dataframe()
        dict_keys = [str(column).strip() for column in forward_df.columns]

        return forward_df.to_bag(index=True).map(row2dict, columns=dict_keys)

    def to_dataframe(self) -> DataFrame:
        """Returns the DataFrame of the data source"""
        return self._df

    def to_forward_dataframe(self) -> DataFrame:
        """
        Adds columns to the DataFrame that are named after the parameter names in the model's forward method.
        The content of these columns is specified in the forward dictionary of the data source yaml file.

        Returns
        -------
        forward_dataframe
            Contains additional columns corresponding to the parameter names of the model's forward method.
        """
        if not self.forward:
            raise ValueError(
                "For a 'forward_dataframe' you need to specify a `ForwardConfiguration`!"
            )

        # This is strictly a shallow copy of the underlying computational graph
        forward_dataframe = self._df.copy()

        forward_dataframe["label"] = (
            forward_dataframe[self.forward.label]
            .astype(str)
            .apply(self.forward.sanitize_label, meta=("label", "object"))
        )
        self._add_forward_token_columns(forward_dataframe)
        # TODO: Remove rows that contain an empty label or empty tokens!!
        #       Not so straight forward: what if record 1 is partially empty, does it produce empty TextFields??

        return forward_dataframe

    def _add_forward_token_columns(self, forward_dataframe: DataFrame):
        """Helper function to add the forward token parameters for the model's forward method"""
        for forward_token_name, data_column_names in self.forward.tokens.items():
            # convert str to list, otherwise the axis=1 raises an error with the returned pd.Series in the next line
            if isinstance(data_column_names, str):
                data_column_names = [data_column_names]

            try:
                forward_dataframe[forward_token_name] = forward_dataframe[
                    data_column_names
                ].apply(
                    lambda x: x.to_dict(), axis=1, meta=(forward_token_name, "object")
                )
            except KeyError as e:
                raise KeyError(
                    e, f"Did not find {data_column_names} in the data source!"
                )
            # if the data source df already has a column with the forward_token_name, it will be replaced!

        return

    @classmethod
    def from_yaml(cls: "DataSource", file_path: str) -> "DataSource":
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

        forward = cfg_dict.pop("forward", None)
        forward_config = (
            ClassificationForwardConfiguration(**forward) if forward else None
        )

        return cls(**cfg_dict, forward=forward_config)


class ClassificationForwardConfiguration(object):
    """
        This ``ClassificationForwardConfiguration`` contains the
        forward transformation of the label and tokens in classification models.

        Parameters
        ----------
        label
            Name of the label column in the data
        target
            (deprecated) Just an alias for label
        tokens
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
        """
        Loads the "metadata_file" (should be called mapping file).
        For now it only allows to map line number -> str.

        Parameters
        ----------
        path
            Path to the metadata (mapping) file

        Returns
        -------
        mapping
            A dict containing the mapping
        """
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
        """Sanitizes the label str, uses a default label (optional), maps the label to a str (optional)"""
        label = label.strip()
        if self.default_label:
            label = label if label else self.default_label
        if self.metadata:
            label = self.metadata.get(label, label)

        return label
