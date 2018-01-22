from os import PathLike

import dill
from allennlp.data import Dataset


def save_to_file(dataset: Dataset, file: str) -> None:
    with open(file, "wb") as file:
        dill.dump(dataset, file)


def load_from_file(file: str) -> Dataset:
    with open(file, "rb") as file:
        return dill.load(file)
