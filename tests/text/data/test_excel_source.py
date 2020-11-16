import os

import pandas as pd
import pytest
from biome.text import Dataset
from datasets.features import Features
from datasets.features import Value

from tests import RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


@pytest.fixture
def features_dict() -> dict:
    """Creating the features dictionary"""
    str_value = Value("string")
    int_value = Value("int64")
    features = Features(
        Notification=int_value, Type=str_value, Plant=int_value, Serial=str_value
    )
    return features


@pytest.fixture
def dataset(features_dict) -> Dataset:
    """
    Creation of dataset
    Excel into dataset via pandas
    """

    file_path = os.path.join(FILES_PATH, "test.xlsx")
    df = pd.read_excel(file_path)

    return Dataset.from_pandas(df, features=features_dict)


def test_read_excel(dataset):
    """Checking that the excel can be read"""

    assert len(dataset) > 0
