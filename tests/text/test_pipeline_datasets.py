import logging
import os

import pandas as pd
import pytest

# fmt: off
from biome.text import (
    Pipeline,
    PipelineConfiguration,
    TrainerConfiguration,
    VocabularyConfiguration,
)
from biome.text.data import DataSource
from biome.text.modules.heads import TextClassificationConfiguration

from allennlp.data import AllennlpDataset, Instance, AllennlpLazyDataset
# fmt: on


@pytest.fixture
def datasource_test(tmp_path) -> DataSource:
    data_file = tmp_path / "classifier.parquet"
    df = pd.DataFrame(
        {
            "text": ["A common text", "This is why you get", "Seriosly?, I'm not sure"],
            "label": ["one", "zero", "zero"],
        }
    )
    df.to_parquet(data_file)

    return DataSource(source=str(data_file))


@pytest.fixture
def datasource_with_partial_mapping(tmp_path) -> DataSource:
    data_file = tmp_path / "classifier.parquet"
    df = pd.DataFrame(
        {
            "another_text": [
                "A common text",
                "This is why you get",
                "Seriosly?, I'm not sure",
            ],
            "label": ["one", "zero", "zero"],
        }
    )
    df.to_parquet(data_file)

    return DataSource(source=str(data_file), mapping={"text": "another_text"})


@pytest.fixture
def pipeline_test() -> Pipeline:
    config = PipelineConfiguration(
        name="test-classifier",
        head=TextClassificationConfiguration(labels=["one", "zero"]),
    )
    return Pipeline.from_config(config)


def test_dataset_creation_with_partial_mapping(
    datasource_with_partial_mapping: DataSource, pipeline_test: Pipeline
):
    df = datasource_with_partial_mapping.to_mapped_dataframe()
    dataset = pipeline_test.create_dataset(datasource_with_partial_mapping)
    assert isinstance(dataset, AllennlpDataset)
    assert len(dataset) == len(df.text)

    for instance in dataset:
        assert isinstance(instance, Instance)
        assert "text" in instance.fields
        assert "label" in instance.fields


def test_datasets_creation(pipeline_test: Pipeline, datasource_test: DataSource):

    df = datasource_test.to_dataframe()
    dataset = pipeline_test.create_dataset(datasource_test)
    assert isinstance(dataset, AllennlpDataset)
    assert len(dataset) == len(df.text)

    for instance in dataset:
        assert isinstance(instance, Instance)
        assert "text" in instance.fields
        assert "label" in instance.fields


def test_lazy_dataset_creation(pipeline_test: Pipeline, datasource_test: DataSource):
    df = datasource_test.to_dataframe()
    dataset = pipeline_test.create_dataset(datasource_test, lazy=True)
    assert isinstance(dataset, AllennlpLazyDataset)
    assert len([x for x in dataset]) == len(df.text)

    for instance in dataset:
        assert isinstance(instance, Instance)
        assert "text" in instance.fields
        assert "label" in instance.fields


def test_training_with_data_bucketing(
    pipeline_test: Pipeline, datasource_test: DataSource, tmp_path: str
):
    lazy_ds = pipeline_test.create_dataset(datasource_test, lazy=True)
    non_lazy_ds = pipeline_test.create_dataset(datasource_test)

    pipeline_test.create_vocabulary(VocabularyConfiguration(sources=[lazy_ds]))

    configuration = TrainerConfiguration(
        data_bucketing=True, batch_size=2, num_epochs=5
    )
    pipeline_test.train(
        output=os.path.join(tmp_path, "output"),
        trainer=configuration,
        training=lazy_ds,
        validation=non_lazy_ds,
    )

    pipeline_test.train(
        output=os.path.join(tmp_path, "output"),
        trainer=configuration,
        training=non_lazy_ds,
        validation=lazy_ds,
    )


def test_training_with_logging(
    pipeline_test: Pipeline, datasource_test: DataSource, tmp_path: str
):
    training = pipeline_test.create_dataset(datasource_test)
    pipeline_test.create_vocabulary(VocabularyConfiguration(sources=[training]))

    configuration = TrainerConfiguration(
        data_bucketing=True, batch_size=2, num_epochs=5
    )
    output_dir = os.path.join(tmp_path, "output")
    pipeline_test.train(
        output=output_dir, trainer=configuration, training=training, quiet=True
    )

    assert os.path.exists(os.path.join(output_dir, "train.log"))
    with open(os.path.join(output_dir, "train.log")) as train_log:
        for line in train_log.readlines():
            assert "allennlp" in line

    assert logging.getLogger("allennlp").level == logging.ERROR
    assert logging.getLogger("biome").level == logging.INFO
