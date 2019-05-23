import glob
from typing import Dict, Optional, Union, List

import dask
import dask.dataframe as dd
import dask.distributed
import pandas as pd
from dask import delayed
from dask.bag import Bag
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from .utils import row2dict


# TODO: The idea is to make the readers a class and define a metaclass that they have to follow.
#       For now, all reader methods have to return a dask.Bag of dicts


def from_csv(path: Union[str, List[str]], columns: List[str] = [], **params) -> Bag:
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
    dataframe = dd.read_csv(path, **params, include_path_column=True)

    columns = (
        [str(column).strip() for column in dataframe.columns]
        if not columns
        else columns
    )
    return dataframe.to_bag(index=True).map(row2dict, columns)


def from_json(path: Union[str, List[str]], **params) -> Bag:
    """
    Creates a dask.Bag of dict objects from a collection of json files

    :param path: The path
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """
    dataframe = dd.read_json(path, **params)

    columns = [str(column).strip() for column in dataframe.columns]
    return dataframe.to_bag(index=True).map(row2dict, columns, path)

def from_parquet(path: Union[str, List[str]], **params) -> Bag:
    """
    Creates a dask.Bag of dict objects from a parquet paths

    :param path: The path
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """
    dataframe = dd.read_parquet(path, **params)

    columns = [str(column).strip() for column in dataframe.columns]
    return dataframe.to_bag(index=True).map(row2dict, columns, path)

def from_excel(path: str, **params) -> Bag:
    """
    Creates a dask.Bag of dict objects from a collection excel files

    :param path: The path
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """
    file_names = glob.glob(path, recursive=True)
    dataframe = dd.from_pandas(
        pd.read_excel(path, **params).fillna(""), npartitions=max(1, len(file_names))
    )

    columns = [str(column).strip() for column in dataframe.columns]

    return dataframe.to_bag(index=True).map(row2dict, columns, path)


def from_elasticsearch(
    query: Optional[Dict] = None,
    npartitions: int = 2,
    client_cls: Optional = None,
    client_kwargs=None,
    source_only=False,
    **kwargs
) -> Bag:
    """Reads documents from Elasticsearch.

    By default, documents are sorted by ``_doc``. For more information see the
    scrolling section in Elasticsearch documentation.

    Parameters
    ----------
    query : dict, optional
        Search query.
    npartitions : int, optional
        Number of partitions, default is 1.
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

    def map_to_source(x: Dict) -> Dict:
        return x["_source"] if source_only else x

    assert (
        npartitions > 1
    ), "Elasticsearch source doesn't work with single partition. Minimum partition size is 2"

    query = query or {}
    # Sorting by _doc is preferred for scrolling.
    query.setdefault("sort", ["_doc"])
    if client_cls is None:
        client_cls = Elasticsearch
    # We load documents in parallel using the scrolling + slicing feature.

    return dask.bag.from_delayed(
        [
            delayed(_elasticsearch_scan)(client_cls, client_kwargs, **scan_kwargs)
            for scan_kwargs in [
                dict(kwargs, query=dict(query, slice=slice))
                for slice in [
                    {"id": idx, "max": npartitions} for idx in range(npartitions)
                ]
            ]
        ]
    ).map(map_to_source)


def _elasticsearch_scan(client_cls, client_kwargs, **params):
    # This method is executed in the worker's process and here we instantiate
    # the ES client as it cannot be serialized.
    client = client_cls(**(client_kwargs or {}))
    return list(scan(client, **params))
