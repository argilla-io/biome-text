import dask.dataframe as df
from dask.bag import Bag

from .utils import row2dict


def from_json(path: str, **params) -> Bag:
    """
    Creates a dask.Bag of dict objects from a collection of json files

    :param path: The path
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """
    dataframe = df.read_json(path, **params)

    columns = [str(column).strip() for column in dataframe.columns]
    return dataframe.to_bag(index=True).map(row2dict, columns, path)
