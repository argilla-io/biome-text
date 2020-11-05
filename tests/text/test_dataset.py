import os

import pandas as pd
from pathlib import Path

from biome.text import Pipeline, Dataset
from tests import RESOURCES_PATH
import pytest


@pytest.fixture(scope="class")
def dataset(tmp_path_factory, request):
    cache_path = tmp_path_factory.mktemp("test_instance_caching")
    data_path = cache_path / "data.csv"
    data_path.write_text("text,label\ncheck,this\nand,that")

    dataset = Dataset.from_csv(paths=str(data_path), cache_dir=str(cache_path))

    # inject to classes decorated with `pytest.mark.usefixtures`
    if request.cls:
        request.cls.dataset = dataset

    return dataset


@pytest.fixture
def default_pipeline_config():
    return {
        "name": "datasets_test",
        "features": {
            "word": {"embedding_dim": 2},
        },
        "head": {"type": "TextClassification", "labels": ["this", "that"]},
    }


def test_load_dataset():
    json_path = os.path.join(RESOURCES_PATH, "data", "dataset_sequence.jsonl")

    with pytest.raises(TypeError):
        Dataset.load_dataset("json", data_files=[json_path])

    ds = Dataset.load_dataset("json", data_files=[json_path], split="train")
    assert len(ds) == 4


def test_from_json():
    json_path = os.path.join(RESOURCES_PATH, "data", "dataset_sequence.jsonl")
    ds = Dataset.from_json(paths=json_path)
    ds2 = Dataset.from_json(paths=[json_path, json_path])

    assert len(ds) == 4
    assert len(ds2) == 8

    json_path = os.path.join(RESOURCES_PATH, "data", "dataset_sequence.json")
    ds = Dataset.from_json(paths=json_path, field="data")

    assert len(ds) == 4


def test_from_csv():
    csv_path = os.path.join(RESOURCES_PATH, "data", "business.cat.2k.valid.csv")
    ds = Dataset.from_csv(paths=csv_path)
    ds2 = Dataset.from_csv(paths=[csv_path, csv_path])

    assert len(ds) == 400
    assert len(ds2) == 800
    assert ds.dataset.column_names == ["label", "text"]


def test_from_pandas():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ds = Dataset.from_pandas(df)

    assert ds.dataset.column_names == ["a", "b"]
    assert ds["a"] == [1, 2, 3]
    assert len(ds) == 3


def test_from_dict():
    ds = Dataset.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})

    assert ds.dataset.column_names == ["a", "b"]
    assert ds["a"] == [1, 2, 3]
    assert len(ds) == 3


def test_to_instances(dataset, default_pipeline_config):
    pl = Pipeline.from_config(default_pipeline_config)
    instances = dataset.to_instances(pl, lazy=False)
    assert len(instances) == 2

    dataset.rename_column_("text", "not_text")
    with pytest.raises(KeyError):
        dataset.to_instances(pl)

    dataset.rename_column_("not_text", "text")
    dataset.rename_column_("label", "not_label")
    # TODO: This should actually raise something ... but not so easy to implement
    dataset.to_instances(pl)


# TODO: this test can go away once we replace our DataSource with Dataset in the dedicated explore test
def test_explore():
    from biome.text import explore

    ds = Dataset.from_json(
        paths=os.path.join(RESOURCES_PATH, "data", "dataset_sequence.jsonl")
    )

    ds.dataset.rename_column_("hypothesis", "text")
    # or to keep the 'hypothesis' column and add the new 'text' column:
    # ds.dataset = ds.dataset.map(lambda x: {"text": x["hypothesis"]})

    labels = list(set(ds["label"]))

    pl = Pipeline.from_config(
        {
            "name": "datasets_test",
            "head": {
                "type": "TextClassification",
                "labels": labels,
            },
        }
    )

    explore.create(pipeline=pl, data_source=ds, batch_size=1, show_explore=False)


@pytest.mark.usefixtures("dataset")
class TestInstanceCaching:
    def uses_cached_instances(self, pipeline_config) -> bool:
        """Checks if the `to_instances` method of the provided pipeline_config uses the cached instances"""
        cache_path = Path(self.dataset.dataset.cache_files[0]["filename"]).parent

        number_of_files_before = len(list(cache_path.iterdir()))
        pipeline = Pipeline.from_config(pipeline_config)
        self.dataset.to_instances(pipeline)
        number_of_files_after = len(list(cache_path.iterdir()))

        return number_of_files_before == number_of_files_after

    def test_same_pipeline(self, default_pipeline_config):
        self.dataset.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        if not self.uses_cached_instances(default_pipeline_config):
            pytest.fail("The same pipelines did not reuse the cached instances!")

    def test_compatible_pipelines(self, default_pipeline_config):
        self.dataset.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        default_pipeline_config["features"]["word"]["embedding_dim"] = 4
        if not self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Compatible pipelines did not reuse the cached instances!")

    def test_incompatible_pipelines(self, default_pipeline_config):
        self.dataset.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        default_pipeline_config["features"]["word"]["lowercase_tokens"] = True
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Incompatible pipelines did reuse the cached instances!")

        default_pipeline_config["features"]["char"] = {
            "embedding_dim": 2,
            "encoder": {"type": "gru", "hidden_size": 2},
        }
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Incompatible pipelines did reuse the cached instances!")

        default_pipeline_config["tokenizer"] = {"max_sequence_length": 10}
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Incompatible pipelines did reuse the cached instances!")

    def test_incompatible_datasets(self, default_pipeline_config):
        self.dataset.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        self.dataset = self.dataset.map(
            lambda x: {}
        )  # this generates a new fingerprint for the dataset
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Incompatible datasets did reuse the cached instances!")

    def test_incompatible_versions(self, default_pipeline_config, monkeypatch):
        from biome.text import dataset

        self.dataset.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        monkeypatch.setattr(dataset, "biome__version__", "mockversion")
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail(
                "Incompatible biome.text versions did reuse the cached instances!"
            )

        monkeypatch.setattr(dataset, "allennlp__version__", "mockversion")
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail(
                "Incompatible allennlp versions did reuse the cached instances!"
            )

        monkeypatch.setattr(dataset, "spacy__version__", "mockversion")
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Incompatible spacy versions did reuse the cached instances!")
