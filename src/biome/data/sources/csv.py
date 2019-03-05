from typing import List

import dask.dataframe as df
from dask.bag import Bag

from .utils import row2dict


def from_csv(path: str, columns: List[str] = [], **params) -> Bag:
    """
    Creates a dask.Bag of dict objects from a collection of csv files

    :param path: The path
    :param columns: The cvs column names
    :param params: extra arguments passed to pandas.read_csv
    :return: dask.bag.Bag
    """
    dataframe = df.read_csv(path, **params, include_path_column=True)

    columns = (
        [str(column).strip() for column in dataframe.columns]
        if not columns
        else columns
    )
    return dataframe.to_bag(index=True).map(row2dict, columns)
