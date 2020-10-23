import os

import pandas as pd

from biome.text import Pipeline, Dataset
from biome.text.configuration import VocabularyConfiguration, TrainerConfiguration
from tests import RESOURCES_PATH
import pytest


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


def test_training_with_dataset():
    # TODO: this test can go away once we replace our DataSource with Dataset
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
            "features": {"word": {"embedding_dim": 2},},
            "head": {"type": "TextClassification", "labels": labels,},
        }
    )

    vocab_config = VocabularyConfiguration(sources=[ds])
    pl.create_vocabulary(vocab_config)

    trainer_config = TrainerConfiguration(
        optimizer={"type": "adam", "lr": 0.01,}, num_epochs=1, cuda_device=-1,
    )

    pl.train(output="output", training=ds, trainer=trainer_config)


# TODO: this test can go away once we replace our DataSource with Dataset
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
            "head": {"type": "TextClassification", "labels": labels, },
        }
    )

    explore.create(pipeline=pl, data_source=ds, batch_size=1, show_explore=False)


