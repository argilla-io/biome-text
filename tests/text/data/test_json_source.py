import os

import pytest
from biome.text import Dataset

from tests import RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


@pytest.fixture
def dataset_source() -> Dataset:
    """Creation of dataset"""

    file_path = os.path.join(FILES_PATH, "dataset_source.jsonl")
    dataset_source = Dataset.from_json(paths=file_path)

    return dataset_source


@pytest.fixture
def dataset_flatten_source() -> Dataset:
    """Creation of the flatten dataset"""

    file_path = os.path.join(FILES_PATH, "to-be-flattened.jsonl")
    dataset_flatten_source = Dataset.from_json(paths=file_path)

    return dataset_flatten_source


@pytest.fixture
def dataset_nested_list() -> Dataset:
    """Creation of the nested-list dataset"""

    file_path = os.path.join(FILES_PATH, "nested-list.jsonl")
    dataset_nested_list = Dataset.from_json(paths=file_path)

    return dataset_nested_list


def test_read_json(dataset_source):
    """Testing JSON reading"""

    assert len(dataset_source) > 0


def test_flatten_json(dataset_flatten_source):
    """Assert that flatten operation divides correctly"""
    dataset_flatten_source.flatten_()

    for c in ["complexData.a", "complexData.b"]:
        assert c in dataset_flatten_source.column_names


def test_flatten_nested_list(dataset_nested_list):
    """Assert that the nested list is processed correctly"""
    dataset_nested_list.flatten_()

    assert len(dataset_nested_list) > 0
