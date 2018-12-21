import atexit
import logging
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import dask
import dask.multiprocessing
from dask.cache import Cache
from dask.distributed import Client
from typing import Dict, Any
from typing import Optional

from biome.spec import ModelDefinition
from biome.spec.utils import to_biome_class

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
from allennlp.common import Params

ENV_DASK_CLUSTER = 'DASK_CLUSTER'
ENV_DASK_CACHE_SIZE = 'DASK_CACHE_SIZE'

DEFAULT_DASK_CACHE_SIZE = 2e9

__logger = logging.getLogger(__name__)


def read_datasource_cfg(cfg: Any) -> Dict:
    try:
        if isinstance(cfg, str):
            with open(cfg) as cfg_file:
                return yaml.load(cfg_file.read())
        if isinstance(cfg, Params):
            return cfg.as_dict()
        if isinstance(cfg, Dict):
            return cfg
    except TypeError or FileNotFoundError:
        raise Exception('Missing configuration {}'.format(cfg))
    except Exception:
        return dict(path=cfg)


def is_biome_model_spec(spec: Dict) -> bool:
    return 'definition' in spec and 'topology' in spec['definition']


def read_definition_from_model_spec(path: str) -> Dict:
    with open(path) as model_file:
        model_data = yaml.load(model_file)
        try:
            model_definition = to_biome_class(data=model_data, klass=ModelDefinition)
            topology = model_definition.topology
            return dict(dataset_reader=topology.pipeline, model=topology.architecture)
        except Exception as e:
            print(e)
            return model_data


def yaml_to_dict(filepath: str):
    with open(filepath) as yaml_content:
        config = yaml.load(yaml_content)
    return config


def read_params_from_file(filepath: str) -> Params:
    try:
        with open(filepath, 'r') as stream:
            params = Params(yaml.load(stream, Loader))
            stream.close()
            return params
    except:
        return Params.from_file(filepath)


def configure_dask_cluster(percentage_use: float = 0.25):
    global dask_client

    dask_cluster = os.environ.get(ENV_DASK_CLUSTER, None)
    dask_cache_size = os.environ.get(ENV_DASK_CACHE_SIZE, DEFAULT_DASK_CACHE_SIZE)
    workers = max(1, round(cpu_count() * percentage_use))

    if dask_cluster:
        dask_client = _dask_client(dask_cluster, dask_cache_size, workers)
    else:
        pool = ThreadPool(workers)
        dask.config.set(pool=pool)
        dask.config.set(num_workers=workers)
        dask.config.set(scheduler='processes')

    __logger.info('Dask configuration:')
    __logger.info(dask.config.config)


@atexit.register
def close_dask_client():
    global dask_client
    try:
        dask_client.close()
    except:
        pass


def _dask_client(dask_cluster: str, cache_size: Optional[int], workers: int) -> Client:
    if cache_size:
        cache = Cache(cache_size)
        cache.register()

    try:
        if dask_cluster == 'local':
            from dask.distributed import Client, LocalCluster
            cluster = LocalCluster(n_workers=workers, threads_per_worker=2)
            return Client(cluster)
        else:
            return dask.distributed.Client(dask_cluster)
    except:
        return dask.distributed.Client()
