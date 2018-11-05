import atexit
import os
from typing import Optional

import dask
import dask.multiprocessing
from dask.cache import Cache
from dask.distributed import Client

ENV_DASK_CLUSTER = 'DASK_CLUSTER'
ENV_DASK_CACHE_SIZE = 'DASK_CACHE_SIZE'

DEFAULT_DASK_CACHE_SIZE = 2e9


def configure_dask_cluster():
    global dask_client

    dask_cluster = os.environ.get(ENV_DASK_CLUSTER, None)
    dask_cache_size = os.environ.get(ENV_DASK_CACHE_SIZE, DEFAULT_DASK_CACHE_SIZE)

    if dask_cluster:
        dask_client = _dask_client(dask_cluster, dask_cache_size)
    else:
        dask.config.set(scheduler='threads')


@atexit.register
def close_dask_client():
    global dask_client
    try:
        dask_client.close()
    except:
        pass


def _dask_client(dask_cluster: str, cache_size: Optional[int]) -> Client:
    if cache_size:
        cache = Cache(cache_size)
        cache.register()

    try:
        return dask.distributed.Client(dask_cluster)
    except:
        return dask.distributed.Client()
