from typing import Dict

from dask.bag import Bag


def file_sink(dataset: Bag, store_options: Dict) -> Bag:
    return dataset.to_textfiles(**store_options)
