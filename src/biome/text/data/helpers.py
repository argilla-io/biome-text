import atexit
import logging
import os.path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import pandas as pd
import yaml
from dask.cache import Cache
from dask.utils import parse_bytes
from distributed import Client, LocalCluster

from biome.text.environment import (
    DEFAULT_DASK_CACHE_SIZE,
    ENV_DASK_CACHE_SIZE,
)

__LOGGER = logging.getLogger(__name__)

__DASK_CLIENT = None


def get_nested_property_from_data(data: Dict, property_key: str) -> Optional[Any]:
    """Search an deep property key in a data dictionary.

    For example, having the data dictionary {"a": {"b": "the value"}}, the call

    >> self.get_nested_property_from_data( {"a": {"b": "the value"}}, "a.b")

    is equivalent to:

    >> data["a"]["b"]


    Parameters
    ----------
    data
        The data dictionary
    property_key
        The (deep) property key

    Returns
    -------

        The property value if found, None otherwise
    """
    if data is None or not isinstance(data, Dict):
        return None

    if property_key in data:
        return data[property_key]

    sep = "."
    splitted_key = property_key.split(sep)

    return get_nested_property_from_data(
        data.get(splitted_key[0]), sep.join(splitted_key[1:])
    )


def configure_dask_cluster(
    address: str = "local", n_workers: int = 1, worker_memory: Union[str, int] = "1GB"
) -> Optional[Client]:
    """Creates a dask client (with a LocalCluster if needed)

    Parameters
    ----------
    address
        The cluster address. If "local" try to connect to a local cluster listening the 8786 port.
        If no cluster listening, creates a new LocalCluster
    n_workers
        The number of cluster workers (only a new "local" cluster generation)
    worker_memory
        The memory reserved for local workers

    Returns
    -------
    A new dask Client

    """
    global __DASK_CLIENT  # pylint: disable=global-statement

    if __DASK_CLIENT:
        return __DASK_CLIENT

    def create_dask_client(
        dask_cluster: str, cache_size: Optional[int], workers: int, worker_mem: int
    ) -> Client:
        if cache_size:
            cache = Cache(cache_size)
            cache.register()

        if dask_cluster != "local":
            return Client(dask_cluster)

        try:
            return Client("localhost:8786", timeout=5)
        except OSError:
            cluster = _create_local_cluster(workers, worker_mem)
            return Client(cluster)

    def _create_local_cluster(workers: int, worker_mem: int):
        """Creates a dask local cluster instance"""
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
            processes=False,
            memory_limit=worker_mem,
            silence_logs=logging.ERROR,
            local_directory=worker_space,
        )
        return cluster

    dask_cluster = address
    dask_cache_size = os.environ.get(ENV_DASK_CACHE_SIZE, DEFAULT_DASK_CACHE_SIZE)

    if isinstance(worker_memory, str):
        worker_memory = parse_bytes(worker_memory)

    __DASK_CLIENT = create_dask_client(
        dask_cluster, dask_cache_size, workers=n_workers, worker_mem=worker_memory
    )

    return __DASK_CLIENT


@atexit.register
def close_dask_client():
    global __DASK_CLIENT  # pylint: disable=global-statement

    try:
        __DASK_CLIENT.close(timeout=10)
        __DASK_CLIENT.cluster.close(timeout=10)
    except Exception as err:  # pylint: disable=broad-except
        __LOGGER.debug(err)
    finally:
        __DASK_CLIENT = None


def extension_from_path(path: Union[str, List[str]]) -> str:
    """Helper method to get file extension

    Parameters
    ----------
    path
        A string or a list of strings.
        If it is a list, the first entry is taken.

    Returns
    -------
    extension
        File extension
    """
    if isinstance(path, str):
        path = [path]

    _, extension = os.path.splitext(path[0])

    return extension.lower()[1:]  # skip first char, which is a dot


def make_paths_relative(yaml_dirname: str, cfg_dict: Dict, path_keys: List[str] = None):
    """Helper method to convert file system paths relative to the yaml config file,
    to paths relative to the current path.

    It will recursively cycle through `cfg_dict` if it is nested.

    Parameters
    ----------
    yaml_dirname
        Dirname to the yaml config file (as obtained by `os.path.dirname`.
    cfg_dict
        The config dictionary extracted from the yaml file.
    path_keys
        If not None, it will only try to modify the `cfg_dict` values corresponding to the `path_keys`.
    """
    for key, value in cfg_dict.items():
        if isinstance(value, dict):
            make_paths_relative(yaml_dirname, value, path_keys)

        if path_keys and key not in path_keys:
            continue

        if is_relative_file_system_path(value):  # returns False if value is not a str
            cfg_dict[key] = os.path.join(yaml_dirname, value)

        # cover lists as well
        if isinstance(value, list):
            cfg_dict[key] = [
                os.path.join(yaml_dirname, path)
                if is_relative_file_system_path(path)
                else path
                for path in value
            ]


def is_relative_file_system_path(string: str) -> bool:
    """Helper method to check if a string is a relative file system path.

    Parameters
    ----------
    string
        The string to be checked.

    Returns
    -------
    bool
        Whether the string is a relative file system path or not.
        If string is not type(str), return False.
    """
    if not isinstance(string, str):
        return False
    # we require the files to have a file name extension ... ¯\_(ツ)_/¯
    if not os.path.isdir(string) and not extension_from_path(string):
        return False
    # check if a domain name
    if string.lower().startswith(
        ("http://", "https://", "ftp://", "sftp://", "s3://", "hdfs://", "gs://")
    ):
        return False
    # check if an absolute path
    if os.path.isabs(string):
        return False
    return True


def _dict_to_list(row: List[Dict]) -> Optional[dict]:
    """ Converts a list of structured data into a dict of list, where every dict key
        is the list aggregation for every key in original dict

        For example:

        l = [{"name": "Frank", "lastName":"Ocean"},{"name":"Oliver","lastName":"Sacks"]
        _dict_to_list(l)
        {"name":["Frank","Oliver"], "lastName":["Ocean", "Sacks"]}
    """
    try:
        for row_i in row:
            if isinstance(row_i, list):
                row = row_i
        return pd.DataFrame(row).to_dict(orient="list")
    except (ValueError, TypeError):
        return None


def _columns_analysis(
    data_frame: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str]]:
    dicts = []
    lists = []
    unmodified = []

    def is_list_of_structured_data(elem) -> bool:
        if isinstance(elem, list):
            for elem_i in elem:
                if isinstance(elem_i, (dict, list)):
                    return True
        return False

    for column in data_frame.columns:
        column_data = data_frame[column].dropna()
        element = column_data.iloc[0] if not column_data.empty else None

        current_list = unmodified
        if isinstance(element, dict):
            current_list = dicts
        elif is_list_of_structured_data(element):
            current_list = lists
        current_list.append(column)

    return dicts, lists, unmodified


def flatten_dask_dataframe(data_frame: dd.DataFrame) -> dd.DataFrame:
    """
    Flatten an dataframe adding nested values as new columns
    and dropping the old ones
    Parameters
    ----------
    data_frame
        The original dask DataFrame

    Returns
    -------

    A new Dataframe with flatten content

    """
    # We must materialize some data for compound the new flatten DataFrame
    meta_flatten = flatten_dataframe(data_frame.head(1))

    def _flatten_stage(data_frame_i: pd.DataFrame) -> pd.DataFrame:
        new_df = flatten_dataframe(data_frame_i)
        for column in new_df.columns:
            # we append the new columns to the original dataframe
            data_frame_i[column] = new_df[column]

        return data_frame_i

    data_frame = data_frame.map_partitions(
        _flatten_stage,
        meta={**data_frame.dtypes.to_dict(), **meta_flatten.dtypes.to_dict()},
    )
    return data_frame[meta_flatten.columns]


def flatten_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    dict_columns, list_columns, unmodified_columns = _columns_analysis(data_frame)

    if len(data_frame.columns) == len(unmodified_columns):
        return data_frame

    dfs = []
    for column in list_columns:
        column_df = pd.DataFrame(
            data=[data for data in data_frame[column].apply(_dict_to_list) if data],
            index=data_frame.index,
        )
        column_df.columns = [
            f"{column}.*.{column_df_column}" for column_df_column in column_df.columns
        ]
        dfs.append(column_df)

    for column in dict_columns:
        column_df = pd.DataFrame(
            data=[data if data else {} for data in data_frame[column]],
            index=data_frame.index,
        )
        column_df.columns = [
            f"{column}.{column_df_column}" for column_df_column in column_df.columns
        ]
        dfs.append(column_df)

    flatten = flatten_dataframe(pd.concat(dfs, axis=1))
    return pd.concat([data_frame[unmodified_columns], flatten], axis=1)


def save_dict_as_yaml(dictionary: dict, path: str, create_dirs: bool = True) -> str:
    """Save a cfg dict to path as yaml

    Parameters
    ----------
    dictionary
        Dictionary to be saved
    path
        Filesystem location where the yaml file will be saved
    create_dirs
        If true, create directories in path.
        If false, throw exception if directories in path do not exist.

    Returns
    -------
    path
        Location of the yaml file
    """
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.isdir(dir_name):
        if not create_dirs:
            raise NotADirectoryError(f"Path '{dir_name}' does not exist.")
        os.makedirs(dir_name)

    with open(path, "w") as yml_file:
        yaml.safe_dump(
            dictionary, yml_file, default_flow_style=False, allow_unicode=True
        )

    return path
