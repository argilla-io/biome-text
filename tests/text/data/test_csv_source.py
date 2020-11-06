import os

import pytest
from biome.text import Dataset

from tests import RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


@pytest.fixture
def dataset() -> Dataset:
    """Creation of dataset"""

    file_path = os.path.join(FILES_PATH, "trailing_coma_in_headers.csv")

    # TODO: erase download mode once the release which includes not caching the datasets is out
    dataset = Dataset.from_csv(paths=file_path, delimiter=";")

    return dataset


def test_reader_csv_with_leading_and_trailing_spaces_in_examples(dataset):
    """Asserting that the csv with ; delimiter is correctly converted into dataset"""

    assert " name" in dataset.column_names
