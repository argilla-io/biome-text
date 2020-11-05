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
    Excel into dataset via pandas
    """

    file_path = os.path.join(FILES_PATH, "test.xlsx")
    df = pd.read_excel(file_path)

    # Dropping problematic columns that has NaN values
    df = df[["Notification", "Notification type"]]
    return Dataset.from_pandas(df)


def test_read_excel(dataset):
    """Checking that the excel can be read"""

    assert len(dataset) > 0
