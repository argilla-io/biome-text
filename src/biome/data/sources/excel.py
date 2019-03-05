import glob

import dask.dataframe as df
import pandas as pd
from dask.bag import Bag

from .utils import row2dict


def from_excel(path: str, **params) -> Bag:
    """
    Creates a dask.Bag of dict objects from a collection excel files

    :param path: The path
    :param params: extra arguments passed to pandas.read_json
    :return: dask.bag.Bag
    """
    file_names = glob.glob(path, recursive=True)
    dataframe = df.from_pandas(
        pd.read_excel(path, **params).fillna(""), npartitions=max(1, len(file_names))
    )

    columns = [str(column).strip() for column in dataframe.columns]

    return dataframe.to_bag(index=True).map(row2dict, columns, path)
