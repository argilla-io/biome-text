import pytest
from biome.text.data import DataSource
from biome.text import Dataset, Pipeline
from biome.text.errors import ActionNotSupportedError
from biome.text.configuration import TrainerConfiguration, VocabularyConfiguration
import pandas as pd


@pytest.fixture
def datasource(tmp_path) -> DataSource:
    data = {
        "text": ["test", "this", "shaight", "good"],
        "label": ["good", "good", "bad", "good"],
    }
    path = tmp_path / "evaluate_test_data.jsonl"
    pd.DataFrame(data).to_json(path, orient="records", lines=True)

    return DataSource(str(path))


@pytest.fixture
def dataset() -> Dataset:
    data = {
        "text": ["test", "this", "shaight", "good"],
        "label": ["good", "good", "bad", "good"],
    }

    return Dataset.from_dict(data)


def test_pipeline_evaluate(dataset, tmp_path):
    labels = list(set(dataset["label"]))
    pl = Pipeline.from_config(
        {
            "name": "test_pipeline_evaluate",
            "head": {"type": "TextClassification", "labels": labels,},
        }
    )

    with pytest.raises(ActionNotSupportedError):
        pl.evaluate(dataset)

    output_path = tmp_path / "test_pipeline_evaluate_output"
    pl.create_vocabulary(VocabularyConfiguration(sources=[dataset]))
    pl.train(
        output=str(output_path),
        training=dataset,
        trainer=TrainerConfiguration(num_epochs=1, batch_size=2),
        quiet=True,
    )

    pl = Pipeline.from_pretrained(str(output_path))
    print(pl.evaluate(dataset))
