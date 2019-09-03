import glob
import logging
import warnings
from typing import Dict, Optional, Union, List

import dask
import dask.dataframe as dd
import dask.distributed
import pandas
import pandas as pd
from dask import delayed
from dask.dataframe import DataFrame
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

_logger = logging.getLogger(__name__)
# TODO: The idea is to make the readers a class and define a metaclass that they have to follow.
#       For now, all reader methods have to return a dask.Bag of dicts


def from_csv(
    path: Union[str, List[str]], columns: List[str] = [], **params
) -> DataFrame:
    """Creates a dask.Bag of dict objects from a collection of csv files

    Parameters
    ----------
    path
        Path to data source
    columns
        Column names of the csv file
    params
        Extra arguments passed on to `pandas.read_csv`

    Returns
    -------
    bag
        A `dask.Bag` of dicts

    """
    return dd.read_csv(path, include_path_column=True, **params)


def from_json(
    path: Union[str, List[str]], flatten: bool = False, **params
) -> DataFrame:
    """
    Creates a dask.Bag of dict objects from a collection of json files

    :param path: The path
    :param flatten: If true (default false), flatten json nested data
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """

    def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        c_dicts = [c for c in df.columns if isinstance(df[c].values[0], dict)]

        if len(c_dicts) == 0:
            return df

        dfs = [df[[c for c in df if c not in c_dicts]]]
        for c in c_dicts:
            c_df = pd.DataFrame([md for md in df[c]])
            c_df.columns = [f"{c}.{cc}" for cc in c_df.columns]
            dfs.append(c_df)

        return flatten_dataframe(pd.concat(dfs, axis=1))

    def json_engine(*args, flatten: bool = False, **kwargs) -> pd.DataFrame:
        df = pd.read_json(*args, **kwargs)
        return flatten_dataframe(df) if flatten else df

    ddf = dd.read_json(path, flatten=flatten, engine=json_engine, **params)
    ddf["path"] = path

    return ddf


def from_parquet(path: Union[str, List[str]], **params) -> DataFrame:
    """
    Creates a dask.Bag of dict objects from a parquet paths

    :param path: The path
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """
    ddf = dd.read_parquet(path, **params)
    ddf["path"] = path

    return ddf


def from_excel(path: str, **params) -> DataFrame:
    """
    Creates a dask.Bag of dict objects from a collection excel files

    :param path: The path
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """
    file_names = glob.glob(path, recursive=True)
    ddf = dd.from_pandas(
        pd.read_excel(path, **params).fillna(""), npartitions=max(1, len(file_names))
    )

    ddf["path"] = path
    return ddf


def from_elasticsearch(
    query: Optional[Dict] = None,
    npartitions: int = 2,
    client_cls: Optional = None,
    client_kwargs=None,
    source_only: Optional[bool] = None,
    **kwargs,
) -> DataFrame:
    """Reads documents from Elasticsearch.

    By default, documents are sorted by ``_doc``. For more information see the
    scrolling section in Elasticsearch documentation.

    Parameters
    ----------
    query : dict, optional
        Search query.
    npartitions : int, optional
        Number of partitions, default is 2.
    client_cls : elasticsearch.Elasticsearch, optional
        Elasticsearch client class.
    client_kwargs : dict, optional
        Elasticsearch client parameters.
    **params
        Additional keyword arguments are passed to the the
        ``elasticsearch.helpers.scan`` function.

    Returns
    -------
    out : List[Delayed]
        A list of ``dask.Delayed`` objects.

    Examples
    --------

    Get all documents in elasticsearch.
    >>> docs = from_elasticsearch()

    Get documents matching a given query.
    >>> query = {"query": {"match_all": {}}}
    >>> docs = from_elasticsearch(query, index="myindex", doc_type="stuff")

    """

    if source_only is not None:
        warnings.warn("source-only field will be removed", DeprecationWarning)

    if npartitions < 2:
        _logger.warning(
            "A minium of 2 partitions is needed for elasticsearch scan slices....Setting partitions number to 2"
        )
        npartitions = 2

    query = query or {}
    # Sorting by _doc is preferred for scrolling.
    query.setdefault("sort", ["_doc"])
    if client_cls is None:
        client_cls = Elasticsearch
    # We load documents in parallel using the scrolling + slicing feature.
    index_scan = [
        delayed(_elasticsearch_scan)(client_cls, client_kwargs, **scan_kwargs)
        for scan_kwargs in [
            dict(kwargs, query=dict(query, slice=slice))
            for slice in [{"id": idx, "max": npartitions} for idx in range(npartitions)]
        ]
    ]

    return dask.dataframe.from_delayed(index_scan)


import flatdict


def _elasticsearch_scan(client_cls, client_kwargs, **params) -> pandas.DataFrame:
    def map_to_source(x: Dict) -> Dict:
        flat = flatdict.FlatDict(
            {**x["_source"], **dict(id=x["_id"], resource=x["_index"])}, delimiter="."
        )
        return dict(flat)

    # This method is executed in the worker's process and here we instantiate
    # the ES client as it cannot be serialized.
    # TODO check empty DataFrame
    client = client_cls(**(client_kwargs or {}))
    return pandas.DataFrame(
        (map_to_source(document) for document in scan(client, **params))
    ).set_index("id")
