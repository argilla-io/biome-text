import logging
from glob import glob
from typing import List, Optional, Union
from urllib.parse import urlparse

import dask.dataframe as dd
import pandas as pd
from dask import delayed
from dask_elk.client import DaskElasticClient

from .helpers import flatten_dask_dataframe, flatten_dataframe

ID = "id"
RESOURCE = "resource"
PATH_COLUMN_NAME = "path"

__LOGGER = logging.getLogger(__name__)
# TODO: The idea is to make the readers a class and define a metaclass that they have to follow.
#       For now, all reader methods have to return a dask.DataFrame. See ElasticsearchDataFrameReader


class DataFrameReader:
    """A base class for read :class:dask.dataframe.DataFrame
    """

    @classmethod
    def read(cls, source: Union[str, List[str]], **kwargs) -> dd.DataFrame:
        """
        Base class method for read the DataSources as a :class:dask.dataframe.DataFrame

        Parameters
        ----------
        source: The source information.
        kwargs: extra arguments passed to read method. Each reader should declare needed arguments

        Returns
        -------

        A :class:dask.dataframe.DataFrame read from source

        """

        raise NotImplementedError


def from_csv(path: Union[str, List[str]], **params) -> dd.DataFrame:
    """Creates a `dask.dataframe.DataFrame` from one or several csv files.
    Includes a "path column".

    Parameters
    ----------
    path
        Path to files
    params
        Extra arguments passed on to `dask.dataframe.read_csv`

    Returns
    -------
    df
        A `dask.DataFrame`

    """
    return dd.read_csv(path, include_path_column=PATH_COLUMN_NAME, **params)


def from_json(
    path: Union[str, List[str]], flatten: bool = True, **params
) -> dd.DataFrame:
    """Creates a `dask.dataframe.DataFrame` from one or several json files.
    Includes a "path column".

    Parameters
    ----------
    path
        Path to files
    flatten
        If true (default false), flatten json nested data
    params
        Extra arguments passed on to `pandas.read_json`

    Returns
    -------
    df
        A `dask.DataFrame`
    """

    def json_engine(*args, flatten: bool = False, **kwargs) -> pd.DataFrame:
        data_frame = pd.read_json(*args, **kwargs)
        return flatten_dataframe(data_frame) if flatten else data_frame

    path_list = _get_file_paths(path)

    dds = []
    for path_name in path_list:
        ddf = dd.read_json(path_name, flatten=flatten, engine=json_engine, **params)
        ddf[PATH_COLUMN_NAME] = path_name
        dds.append(ddf)

    return dd.concat(dds)


def from_parquet(path: Union[str, List[str]], **params) -> dd.DataFrame:
    """Creates a `dask.dataframe.DataFrame` from one or several parquet files.
    Includes a "path column".

    Parameters
    ----------
    path
        Path to files
    params
        Extra arguments passed on to `pandas.read_parquet`

    Returns
    -------
    df
        A `dask.dataframe.DataFrame`
    """
    path_list = _get_file_paths(path)

    dds = []
    for path_name in path_list:
        ddf = dd.read_parquet(path_name, engine="pyarrow", **params)
        ddf[PATH_COLUMN_NAME] = path_name
        dds.append(ddf)

    return dd.concat(dds)


def from_excel(path: Union[str, List[str]], **params) -> dd.DataFrame:
    """Creates a `dask.dataframe.DataFrame` from one or several excel files.
    Includes a "path column".

    Parameters
    ----------
    path
        Path to files
    params
        Extra arguments passed on to `pandas.read_excel`

    Returns
    -------
    df
        A `dask.dataframe.DataFrame`
    """
    path_list = _get_file_paths(path)

    dds = []
    for path_name in path_list:
        parts = delayed(pd.read_excel)(path_name, **params)
        data_frame = dd.from_delayed(parts).fillna("")
        data_frame[PATH_COLUMN_NAME] = path_name
        dds.append(data_frame)

    return dd.concat(dds)


def _get_file_paths(paths: Union[str, List[str]]) -> List[str]:
    """Return a list of path names that match the path names in paths.
    The path names can contain shell-style wildcards.

    Parameters
    ----------
    paths
        A path name or a list of path names. These path names can contain wildcards.

    Returns
    -------
    list_of_paths
        A list of path names.
    """
    if isinstance(paths, str):
        path_list = glob(paths)
        return path_list if path_list else [paths]
    path_lists = [_get_file_paths(path) for path in paths]

    # flatten the list of lists
    return [path for sublist in path_lists for path in sublist]


class ElasticsearchDataFrameReader(DataFrameReader):
    """
        Read a :class:dask.dataframe.DataFrame from a elasticsearch index
    """

    SOURCE_TYPE = "elasticsearch"
    __ELASTIC_ID_FIELD = "_id"

    @classmethod
    def read(  # pylint: disable=arguments-differ
        cls,
        source: Union[str, List[str]],
        index: str,
        doc_type: str = "_doc",
        query: Optional[dict] = None,
        es_host: str = "http://localhost:9200",
        flatten_content: bool = False,
        **kwargs,
    ) -> dd.DataFrame:
        """
        Creates a :class:dask.dataframe.DataFrame from a elasticsearch indexes

        Parameters
        ----------
        source
            The source param must match with :class:ElasticsearchDataFrameReader.SOURCE_TYPE
        es_host
            The elasticsearch host url (default to "http://localhost:9200")
        index
            The elasticsearch index
        doc_type
            The elasticsearch document type (default to "_doc")
        query
            The index query applied for extract the data
        flatten_content
            If True, applies a flatten to all nested data. It may take time to apply this flatten, so
            is deactivate by default.
        kwargs
            Extra arguments passed to base search method

        Returns
        -------
        A :class:dask.dataframe.DataFrame with index query results

        """
        if source != cls.SOURCE_TYPE:
            raise TypeError(f"{source} is not processable with {cls.__class__} reader")

        url = urlparse(es_host)
        client = DaskElasticClient(host=url.hostname, port=url.port, wan_only=True)
        data_frame = client.read(
            query=query, index=index, doc_type=doc_type, **kwargs
        ).persist()

        if flatten_content:
            data_frame = flatten_dask_dataframe(data_frame)
        data_frame[RESOURCE] = f"{index}/{doc_type}"

        if cls.__ELASTIC_ID_FIELD in data_frame.columns:
            data_frame = data_frame.rename(columns={cls.__ELASTIC_ID_FIELD: ID})

        return data_frame
