from typing import Iterable, List
import ujson as json

from dask.bag import Bag
import dask.bag as db
import dask.dataframe as df
from flatten_json import flatten


def from_csv(path: str, columns: List[str] = list(), **params) -> Bag:
    dataframe = df.read_csv(path, **params)
    if not columns:
        columns = [column.strip() for column in dataframe.columns]
    return dataframe.to_bag(index=False).map(lambda row: dict(zip(columns, row)))


def from_json(path: str, **params) -> Bag:
    return db.read_text(path, **params) \
        .map(json.loads) \
        .map(lambda dict: flatten(dict, separator='.'))
