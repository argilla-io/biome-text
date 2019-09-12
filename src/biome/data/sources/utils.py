from typing import Tuple, List, Optional, Dict, Any, Union
import os.path
import pandas as pd

ID = "id"
RESOURCE = "resource"

_DASK_PATH_COLUMN_NAME = "path"


def row2dict(
    row: Tuple, columns: List[str], default_path: Optional[str] = None
) -> Dict[str, Any]:

    """ Convert a pandas row into a dict object """
    id = row[0]
    data = row[1:]

    # For duplicated column names, pandas append a index prefix with dots '.' We prevent
    # index failures by replacing for '_'
    sanitized_columns = [column.replace(".", "_") for column in columns]
    data = dict([(ID, id)] + list(zip(sanitized_columns, data)))

    # DataFrame.read_csv allows include path column called `path`
    data[RESOURCE] = data.get(
        RESOURCE, data.get(_DASK_PATH_COLUMN_NAME, str(default_path))
    )

    return data


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


def make_paths_relative(
    yaml_dirname: str, cfg_dict: Dict, path_keys: Union[str, List[str]] = None
):
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
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            make_paths_relative(yaml_dirname, v)

        if path_keys and k not in path_keys:
            continue

        if is_relative_file_system_path(v):  # returns False if v is not a str
            cfg_dict[k] = os.path.join(yaml_dirname, v)

        # cover lists as well
        if isinstance(v, list):
            cfg_dict[k] = [
                os.path.join(yaml_dirname, path)
                if is_relative_file_system_path(path)
                else path
                for path in v
            ]

        pass


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
    if not extension_from_path(string):
        return False
    # check if a domain name
    if string.lower().startswith(("http", "ftp")):
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
        for r in row:
            if isinstance(r, list):
                row = r
        return pd.DataFrame(row).to_dict(orient="list")
    except (ValueError, TypeError):
        return None


def _columns_analysis(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    dicts = []
    lists = []
    unmodified = []

    def is_dict(e) -> bool:
        return isinstance(e, dict)

    def is_list_of_structured_data(e) -> bool:
        if isinstance(e, list):
            for x in e:
                if isinstance(x, (dict, list)):
                    return True
        return False

    for c in df.columns:
        column_data = df[c].dropna()
        element = column_data.iloc[0] if not column_data.empty else None

        current_list = unmodified
        if is_dict(element):
            current_list = dicts
        elif is_list_of_structured_data(element):
            current_list = lists
        current_list.append(c)

    return dicts, lists, unmodified


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    c_dicts, c_lists, c_unmodified = _columns_analysis(df)

    if len(df.columns) == len(c_unmodified):
        return df

    dfs = []
    for c in c_lists:
        c_df = pd.DataFrame([d for d in df[c].apply(_dict_to_list) if d])
        c_df.columns = [f"{c}.*.{cc}" for cc in c_df.columns]
        dfs.append(c_df)

    for c in c_dicts:
        c_df = pd.DataFrame([md if md else {} for md in df[c]])
        c_df.columns = [f"{c}.{cc}" for cc in c_df.columns]
        dfs.append(c_df)

    flatten = flatten_dataframe(pd.concat(dfs, axis=1))
    return pd.concat([df[c_unmodified], flatten], axis=1)
