import pytest
from biome.text import Dataset
from biome.text import explore
from biome.text import Pipeline
from elasticsearch import Elasticsearch


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
    explore_id = explore.create(
        pl, dataset_without_mapping, explore_id="mock", show_explore=False
    )
    elastic = Elasticsearch()
    explore_meta = elastic.get(index=".biome", id=explore_id)["_source"]
    assert pl.config.as_dict() == explore_meta["pipeline_config"]


def test_explore_with_no_mapping_in_ds(monkeypatch, dataset_without_mapping):
    """Test the creation of the ElasticSearch instance without mapping"""

    def _explore(explore_id, pipeline, dataset, options, es_dao):
        return dataset.to_mapped_dataframe().compute()

    def _show(explore_id, es_host):
        return None

    monkeypatch.setattr(explore, "_explore", _explore)
    monkeypatch.setattr(explore, "show", _show)

    pl = Pipeline.from_config(
        {"name": "test", "head": {"type": "TextClassification", "labels": ["a"]}}
    )
    explore_id = explore.create(pl, dataset_without_mapping)
    assert len(explore_id) > 0
