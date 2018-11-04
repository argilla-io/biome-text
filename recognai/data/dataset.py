from os import PathLike

import dill
from typing import Iterable

from allennlp.data import Instance


def save_to_file(dataset: Iterable[Instance], file: str) -> None:
    with open(file, "wb") as file:
        dill.dump(dataset, file)


def load_from_file(file: str) -> Iterable[Instance]:
    with open(file, "rb") as file:
        return dill.load(file)
