import pytest
from elasticsearch import Elasticsearch

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import explore


@pytest.fixture
def dataset() -> Dataset:
    """Creating the dataset"""
    data = {"text": ["This is a simple test"], "label": ["a"]}
    return Dataset.from_dict(data)


def test_explore_creation(dataset):
    """Test the creation of an index in elasticsearch"""

    pl = Pipeline.from_config(
        {"name": "test", "head": {"type": "TextClassification", "labels": ["a"]}}
    )
    explore_id = explore.create(pl, dataset, explore_id="mock", show_explore=False)
    elastic = Elasticsearch()
    explore_meta = elastic.get(index=".biome", id=explore_id)["_source"]
    assert pl.config.as_dict() == explore_meta["pipeline_config"]
    assert len(explore_id) > 0
