# -*- coding: utf-8 -*-
"""An Elasticsearch reader for Dask.
"""
from typing import Dict, Optional

import dask
import dask.distributed
from dask import bag
from dask import delayed
from dask.bag import Bag
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan


def __elasticsearch_scan(client_cls, client_kwargs, **params):
    # This method is executed in the worker's process and here we instantiate
    # the ES client as it cannot be serialized.
    client = client_cls(**(client_kwargs or {}))
    return list(scan(client, **params))


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
            delayed(__elasticsearch_scan)(client_cls, client_kwargs, **scan_kwargs)
            for scan_kwargs in [
                dict(kwargs, query=dict(query, slice=slice))
                for slice in [
                    {"id": idx, "max": npartitions} for idx in range(npartitions)
                ]
            ]
        ]
    ).map(map_to_source)
