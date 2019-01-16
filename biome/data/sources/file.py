from typing import Iterable, List, Dict, Callable
import ujson as json

from dask.bag import Bag
import dask.bag as db
import dask.dataframe as df
from flatten_json import flatten


def from_csv(path: str, columns: List[str] = [], **params) -> Bag:
    dataframe = df.read_csv(path, na_filter=False, **params)  # na_filter: do not convert any values to np.nan
    if not columns:
        columns = [str(column).strip() for column in dataframe.columns]
    return dataframe.to_bag(index=False).map(lambda row: dict(zip(columns, row)))


def from_json(path: str, **params) -> Bag:
    return db.read_text(path, **params) \
        .map(json.loads) \
        .map(lambda dict: flatten(dict, separator='.'))


def from_raw_text(path: str, line_processor: Callable[[str], Dict] = None, **params) -> Bag:
    bag = db.read_text(path, **params)
    return bag.map(line_processor) if line_processor else bag
