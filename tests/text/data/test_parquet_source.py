import os

import pandas as pd
import pytest
from biome.text import Dataset

from tests import RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


@pytest.fixture
def dataset() -> Dataset:
    """
    Creation of dataset
    Parquet into dataset via pandas
    """

    file_path = os.path.join(FILES_PATH, "test.parquet")
    df = pd.read_parquet(file_path)

    dataset = Dataset.from_pandas(df)
    return dataset


def test_read_parquet(dataset):
    """Asserting that the parquet with  is correctly converted into dataset"""

    assert "reviewerID" in dataset.column_names
