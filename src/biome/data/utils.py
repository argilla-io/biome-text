import atexit
import logging
import os
import re
from multiprocessing.pool import ThreadPool
from typing import Dict, Any
from typing import Optional

import dask
import dask.multiprocessing
from dask.cache import Cache
from dask.distributed import Client

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import yaml

ENV_DASK_CLUSTER = "DASK_CLUSTER"
ENV_DASK_CACHE_SIZE = "DASK_CACHE_SIZE"
DEFAULT_DASK_CACHE_SIZE = 2e9

ENV_ES_HOSTS = "ES_HOSTS"

__logger = logging.getLogger(__name__)


def get_nested_property_from_data(data: Dict, property_key: str) -> Optional[Any]:
    if data is None or not isinstance(data, Dict):
        return None
    if property_key in data:
        return data[property_key]
    else:
        sep = "."
        splitted_key = property_key.split(sep)
        return get_nested_property_from_data(
            data.get(splitted_key[0]), sep.join(splitted_key[1:])
        )


def is_document(source_config: Dict[str, Any]) -> bool:
    return source_config.get("format", "raw") == "raw"


def default_elasticsearch_sink(
    source_config: str, binary_path: str, es_batch_size: int
):
    def sanizite_index(index_name: str) -> str:
        return re.sub("\W", "_", index_name)

    file_name = os.path.basename(source_config)
    model_name = os.path.dirname(binary_path)
    if not model_name:
        model_name = binary_path

    return dict(
        index=sanizite_index(
            "prediction {} with {}".format(file_name, model_name).lower()
        ),
        index_recreate=True,
        type="_doc",
        es_hosts=os.getenv(ENV_ES_HOSTS, "http://localhost:9200"),
        es_batch_size=es_batch_size,
    )


def read_datasource_cfg(cfg: Any) -> Dict:
    try:
        if isinstance(cfg, str):
            with open(cfg) as cfg_file:
                return yaml.load(cfg_file.read())
        if isinstance(cfg, Dict):
            return cfg
    except Exception as e:
        __logger.debug(e)
        return dict(path=cfg)


def yaml_to_dict(filepath: str):
    with open(filepath) as yaml_content:
        config = yaml.load(yaml_content)
    return config


def read_params_from_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as stream:
        return yaml.load(stream, Loader)


def configure_dask_cluster(n_workers: int = 1, worker_memory: int = 3e9):
    def create_dask_client(
        dask_cluster: str, cache_size: Optional[int], workers: int
    ) -> Client:
        if cache_size:
            cache = Cache(cache_size)
            cache.register()

        try:
            if dask_cluster == "local":
                from dask.distributed import Client, LocalCluster

                dask.config.set(
                    {
                        "distributed.worker.memory": dict(
                            target=0.95, spill=False, pause=False, terminate=False
                        )
                    }
                )
                cluster = LocalCluster(
                    n_workers=0, threads_per_worker=1, memory_limit=worker_memory
                )
                cluster.scale_up(workers)
                return Client(cluster)
            else:
                return dask.distributed.Client(dask_cluster)
        except:
            return dask.distributed.Client()

    global dask_client
    # import pandas as pd
    # pd.options.mode.chained_assignment = None

    dask_cluster = os.environ.get(ENV_DASK_CLUSTER, None)
    dask_cache_size = os.environ.get(ENV_DASK_CACHE_SIZE, DEFAULT_DASK_CACHE_SIZE)

    if dask_cluster:
        dask_client = create_dask_client(
            dask_cluster, dask_cache_size, workers=n_workers
        )
    else:
        pool = ThreadPool(n_workers)
        dask.config.set(pool=pool)
        dask.config.set(num_workers=n_workers)
        dask.config.set(scheduler="processes")

    __logger.info("Dask configuration:")
    __logger.info(dask.config.config)


@atexit.register
def close_dask_client():
    global dask_client

    try:
        dask_client.close()
    except:
        pass
