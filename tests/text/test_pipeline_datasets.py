import pandas as pd
import pytest
from allennlp.data import AllennlpDataset, Instance, AllennlpLazyDataset

from biome.text import Pipeline, PipelineConfiguration
from biome.text.data import DataSource
from biome.text.modules.heads import TextClassificationConfiguration


@pytest.fixture
def datasource_test(tmp_path) -> DataSource:
    data_file = tmp_path / "classifier.parquet"
    df = pd.DataFrame(
        {"text": ["A common text", "This is why you get", "Seriosly?, I'm not sure"], "label": ["one", "zero", "zero"]}
    )
    df.to_parquet(data_file)

    return DataSource(source=str(data_file))


@pytest.fixture
def pipeline_test() -> Pipeline:
    config = PipelineConfiguration(
        name="test-classifier",
        head=TextClassificationConfiguration(labels=["one", "zero"])
    )
    return Pipeline.from_config(config)


def test_datasets_creation(pipeline_test: Pipeline, datasource_test: DataSource):

    df = datasource_test.to_dataframe()
    dataset = pipeline_test.create_dataset(datasource_test)
    assert isinstance(dataset, AllennlpDataset)
    assert len(dataset) == len(df.text)

    for instance in dataset:
        assert isinstance(instance, Instance)


def test_lazy_dataset_creation(pipeline_test: Pipeline, datasource_test:DataSource):
    df = datasource_test.to_dataframe()
    dataset = pipeline_test.create_dataset(datasource_test, lazy=True)
    assert isinstance(dataset, AllennlpLazyDataset)
    assert len([x for x in dataset]) == len(df.text)

    for instance in dataset:
        assert isinstance(instance, Instance)
        assert "text" in instance.fields
        assert "label" in instance.fields
