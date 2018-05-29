# -*- coding: utf-8 -*-
"""An Elasticsearch reader for Dask.
"""
import logging
from typing import Dict, Optional

import dask
import dask.distributed
from dask import delayed
from dask import bag
from dask.bag import Bag
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan


def __elasticsearch_scan(client_cls, client_kwargs, **params):
    # This method is executed in the worker's process and here we instantiate
    # the ES client as it cannot be serialized.
    client = client_cls(**(client_kwargs or {}))
    return list(scan(client, **params))


def from_elasticsearch(query: Optional[Dict] = None, npartitions: int = 8, client_cls: Optional = None,
                       client_kwargs=None,
                       source_only=False, **kwargs) -> Bag:
    """Reads documents from Elasticsearch.

    By default, documents are sorted by ``_doc``. For more information see the
    scrolling section in Elasticsearch documentation.

    Parameters
    ----------
    query : dict, optional
        Search query.
    npartitions : int, optional
        Number of partitions, default is 8.
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

    >>> docs = dask.bag.from_delayed(from_elasticsearch())

    Get documents matching a given query.

    >>> query = {"query": {"match_all": {}}}
    >>> docs = dask.bag.from_delayed(from_elasticsearch(query, index="myindex", doc_type="stuff"))

    """

    def map_to_source(x: Dict) -> Dict:
        return x['_source'] if source_only else x

    query = query or {}
    # Sorting by _doc is preferred for scrolling.
    query.setdefault('sort', ['_doc'])
    if client_cls is None:
        client_cls = Elasticsearch
    # We load documents in parallel using the scrolling + slicing feature.

    return dask.bag.from_delayed(
        [delayed(__elasticsearch_scan)(client_cls, client_kwargs, **scan_kwargs)
         for scan_kwargs in [dict(kwargs, query=dict(query, slice=slice))
                             for slice in
                             [{'id': idx, 'max': npartitions} for idx in range(npartitions)]]]) \
        .map(map_to_source)


if __name__ == '__main__':
    cluster = dask.distributed.LocalCluster(n_workers=1, threads_per_worker=1, silence_logs=logging.DEBUG)
    client = dask.distributed.Client(cluster)

    data = from_elasticsearch(npartitions=8, client_kwargs={'hosts': 'http://localhost:9200'}, index='gourmet-food',
                              source_only=True).persist()
    print(data.count().compute())
    for example in data:
        print(example)

    cluster.close()