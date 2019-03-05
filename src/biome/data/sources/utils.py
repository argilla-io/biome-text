from typing import Tuple, List, Optional, Dict, Any


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
    data[RESOURCE] = data.get(_DASK_PATH_COLUMN_NAME, str(default_path))

    return data
