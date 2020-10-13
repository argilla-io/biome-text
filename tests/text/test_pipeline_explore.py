from biome.text import explore
from biome.text.data import DataSource
import pytest
import pandas as pd
from biome.text import Pipeline


@pytest.fixture
def data_source_without_mapping(tmp_path) -> DataSource:
    data_file = tmp_path / "train.json"
    df = pd.DataFrame({"text": ["This is a simple test"], "label": ["a"]})
    df.to_json(data_file, lines=True, orient="records")

    return DataSource(
        source=str(data_file), flatten=False, lines=True, orient="records"
    )


def test_explore_with_no_mapping_in_ds(monkeypatch, data_source_without_mapping):
    def _explore(explore_id, pipeline, data_source, options, es_dao):
        return data_source.to_mapped_dataframe().compute()

    def _show(explore_id, es_host):
        return None

    monkeypatch.setattr(explore, "_explore", _explore)
    monkeypatch.setattr(explore, "show", _show)

    pl = Pipeline.from_config(
        {"name": "test", "head": {"type": "TextClassification", "labels": ["a"]}}
    )
    explore_id = explore.create(pl, data_source_without_mapping)
    assert len(explore_id) > 0
    assert data_source_without_mapping.mapping == {"label": "label", "text": "text"}
