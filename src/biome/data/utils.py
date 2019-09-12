import atexit
import logging
import os
import re
import tempfile
from multiprocessing.pool import ThreadPool

import dask
import dask.multiprocessing
from dask.cache import Cache
from dask.distributed import Client
from dask.utils import parse_bytes
from typing import Dict, Any, Union
from typing import Optional

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import yaml
from dask.distributed import Client, LocalCluster

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
) -> Dict:
    def sanizite_index(index_name: str) -> str:
        return re.sub(r"\W", "_", index_name)

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


def yaml_to_dict(filepath: str):
    with open(filepath) as yaml_content:
        config = yaml.safe_load(yaml_content)
    return config


def read_params_from_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as stream:
        return yaml.load(stream, Loader)


def configure_dask_cluster(
    address: str = "local", n_workers: int = 1, worker_memory: Union[str, int] = "1GB"
) -> Optional[Client]:
    global dask_client
    try:
        if dask_client:
            return dask_client
    except:
        pass

    def create_dask_client(
        dask_cluster: str, cache_size: Optional[int], workers: int, worker_mem: int
    ) -> Client:
        if cache_size:
            cache = Cache(cache_size)
            cache.register()

        if dask_cluster == "local":
            try:
                return Client("localhost:8786", timeout=5)
            except OSError:
                dask.config.set(
                    {
                        "distributed.worker.memory": dict(
                            target=0.95, spill=False, pause=False, terminate=False
                        )
                    }
                )

                dask_data = os.path.join(os.getcwd(), ".dask")
                os.makedirs(dask_data, exist_ok=True)

                worker_space = tempfile.mkdtemp(dir=dask_data)
                cluster = LocalCluster(
                    n_workers=workers,
                    threads_per_worker=1,
                    asynchronous=False,
                    scheduler_port=8786,  # TODO configurable
                    memory_limit=worker_mem,
                    silence_logs=logging.ERROR,
                    local_dir=worker_space,
                )
                return Client(cluster)
        else:
            return dask.distributed.Client(dask_cluster)

    # import pandas as pd
    # pd.options.mode.chained_assignment = None

    dask_cluster = address
    dask_cache_size = os.environ.get(ENV_DASK_CACHE_SIZE, DEFAULT_DASK_CACHE_SIZE)

    if dask_cluster:
        if isinstance(worker_memory, str):
            worker_memory = parse_bytes(worker_memory)

        return create_dask_client(
            dask_cluster, dask_cache_size, workers=n_workers, worker_mem=worker_memory
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
