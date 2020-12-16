import time
from pathlib import Path

import pandas as pd
import pytest
from allennlp.data import AllennlpDataset
from allennlp.data import AllennlpLazyDataset
from allennlp.data import Instance
from datasets.features import Features
from datasets.features import Value
from elasticsearch import Elasticsearch

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import explore


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
        "features": {"word": {"embedding_dim": 2}},
        "head": {"type": "TextClassification", "labels": ["this", "that"]},
    }


def test_load_dataset(resources_data_path):
    json_path = str(resources_data_path / "dataset_sequence.jsonl")

    with pytest.raises(TypeError):
        Dataset.load_dataset("json", data_files=[json_path])

    ds = Dataset.load_dataset("json", data_files=[json_path], split="train")
    assert len(ds) == 4


def test_from_json(resources_data_path):
    json_path = str(resources_data_path / "dataset_sequence.jsonl")
    ds = Dataset.from_json(paths=json_path)
    ds2 = Dataset.from_json(paths=[json_path, json_path])

    assert len(ds) == 4
    assert len(ds2) == 8

    json_path = str(resources_data_path / "dataset_sequence.json")
    ds = Dataset.from_json(paths=json_path, field="data")

    assert len(ds) == 4


def test_flatten_json(resources_data_path):
    """Showcases the behavior of Dataset.flatten_"""
    file_path = str(resources_data_path / "to-be-flattened.jsonl")
    dataset_flatten_source = Dataset.from_json(paths=file_path)
    dataset_flatten_source.flatten_()

    for c in ["complexData.a", "complexData.b"]:
        assert c in dataset_flatten_source.column_names

    file_path = str(resources_data_path / "nested-list.jsonl")
    dataset_nested_list = Dataset.from_json(paths=file_path)
    dataset_nested_list.flatten_()

    assert len(dataset_nested_list) == 1
    assert dataset_nested_list.column_names == ["classification"]


def test_from_csv(resources_data_path):
    csv_path = str(resources_data_path / "business.cat.2k.valid.csv")
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


def test_from_excel_file(resources_data_path):
    """This only shows an example of how one could read in an excel file"""
    str_value = Value("string")
    int_value = Value("int64")
    features = Features(
        Notification=int_value, Type=str_value, Plant=int_value, Serial=str_value
    )

    file_path = resources_data_path / "test.xlsx"
    df = pd.read_excel(file_path)

    dataset = Dataset.from_pandas(df, features=features)

    assert len(dataset) > 0


def test_from_parquet_file(resources_data_path):
    """This only shows an example of how one could read in a parquet file"""
    file_path = resources_data_path / "test.parquet"
    df = pd.read_parquet(file_path)
    dataset = Dataset.from_pandas(df)

    assert "reviewerID" in dataset.column_names


def __wait_for_index_creation__(es_client: Elasticsearch, es_index: str):
    retries = 0
    max_retries = 3

    while retries < max_retries and not es_client.count(index=es_index)["count"]:
        retries += 1
        time.sleep(retries)

    if retries >= max_retries:
        raise Exception(
            f"Max retries reached. Index {es_index} could not be properly created"
        )


def test_from_elasticsearch(dataset, default_pipeline_config):
    pipeline = Pipeline.from_config(default_pipeline_config)
    es_index = explore.create(
        pipeline, dataset, explore_id="test_index", show_explore=False
    )
    es_client = Elasticsearch()
    __wait_for_index_creation__(es_client, es_index)
    ds = Dataset.from_elasticsearch(
        es_client, index=es_index, query={"query": {"match_all": {}}}
    )

    assert len(ds) == len(dataset)
    for key in ["_id", "_index", "_type", "_score"]:
        assert key in ds.column_names

    ds = Dataset.from_elasticsearch(
        es_client,
        index=es_index,
        query={"query": {"exists": {"field": "not_found.field"}}},
    )
    assert len(ds) == 0

    ds = Dataset.from_elasticsearch(
        es_client, index=es_index, fields=["label", "text", "_id"]
    )
    assert len(ds) == len(dataset)
    assert "label" in ds.column_names
    assert "text" in ds.column_names
    assert "_id" in ds.column_names
    assert "prediction" not in ds.column_names


def test_fail_using_reserved_words():
    # This issue was reported and maybe gets resolved: https://github.com/huggingface/datasets/issues/1110

    ds = Dataset.from_dict(
        {
            "a": [{"a": 1, "b": "two"} for _ in range(0, 100)],
            "_type": ["whatever" for _ in range(0, 100)],
        }
    )
    split = ds.train_test_split()
    new_ds = Dataset.from_datasets(split.values())

    assert len(new_ds) == len(ds)

    ds = ds.map(
        lambda example: {
            "new_field": {"c": str(example["a"]["a"]), "d": f"this {example['a']['b']}"}
        }
    )
    split = ds.train_test_split()
    with pytest.raises(TypeError):
        Dataset.from_datasets(split.values())


def test_to_instances(dataset, default_pipeline_config):
    pl = Pipeline.from_config(default_pipeline_config)

    instances = dataset.to_instances(pl, lazy=True)
    assert isinstance(instances, AllennlpLazyDataset)

    instances = dataset.to_instances(pl)
    assert isinstance(instances, AllennlpDataset)

    assert len(instances) == len(dataset)
    for instance in instances:
        assert isinstance(instance, Instance)
        assert "text" in instance.fields
        assert "label" in instance.fields

    # TODO: A missing label column should actually raise something ... but not so easy to implement
    dataset.rename_column_("label", "not_label")
    dataset.to_instances(pl)

    # Missing input column should raise a KeyError
    dataset.rename_column_("text", "not_text")
    with pytest.raises(KeyError):
        dataset.to_instances(pl)


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
        self.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        if not self.uses_cached_instances(default_pipeline_config):
            pytest.fail("The same pipelines did not reuse the cached instances!")

    def test_compatible_pipelines(self, default_pipeline_config):
        self.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        default_pipeline_config["features"]["word"]["embedding_dim"] = 4
        if not self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Compatible pipelines did not reuse the cached instances!")

    def test_incompatible_pipelines(self, default_pipeline_config):
        self.dataset.cleanup_cache_files()
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
        self.dataset.cleanup_cache_files()
        assert not self.uses_cached_instances(default_pipeline_config)

        self.dataset = self.dataset.map(
            lambda x: {}
        )  # this generates a new fingerprint for the dataset
        if self.uses_cached_instances(default_pipeline_config):
            pytest.fail("Incompatible datasets did reuse the cached instances!")

    def test_incompatible_versions(self, default_pipeline_config, monkeypatch):
        from biome.text import dataset

        self.dataset.cleanup_cache_files()
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
