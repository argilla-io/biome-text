from biome.text import _helpers
from biome.text.data import DataSource
import pytest
import pandas as pd
from biome.text import Pipeline


@pytest.fixture
def data_source_without_mapping(tmp_path) -> DataSource:
    data_file = tmp_path / "train.json"
    df = pd.DataFrame({"text": ["This is a simple test"], "label": ["a"],})
    df.to_json(data_file, lines=True, orient="records")

    return DataSource(
        source=str(data_file), flatten=False, lines=True, orient="records"
    )


def test_explore_with_no_mapping_in_ds(monkeypatch, data_source_without_mapping):
    def _explore(self, ds, config, es_config):
        return ds.to_mapped_dataframe().compute()

    def _show_explore(es_config):
        return None

    monkeypatch.setattr(_helpers, "_explore", _explore)
    monkeypatch.setattr(_helpers, "_show_explore", _show_explore)

    pl = Pipeline.from_config(
        {"name": "test", "head": {"type": "TextClassification", "labels": ["a"]}}
    )
    pl.explore(data_source_without_mapping)
    assert data_source_without_mapping.mapping == {"label": "label", "text": "text"}
