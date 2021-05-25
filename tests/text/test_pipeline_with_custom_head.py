import os
from pathlib import Path
from tempfile import mkdtemp

import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import PipelineConfiguration
from biome.text.configuration import FeaturesConfiguration
from biome.text.configuration import VocabularyConfiguration
from biome.text.modules.heads import TaskHeadConfiguration
from biome.text.modules.heads import TextClassification


class MyCustomHead(TextClassification):
    """Just a head renaming the original TextClassification head"""

    pass


@pytest.fixture
def training_dataset() -> Dataset:
    """Creates the training dataset and gives the structure"""
    resources_path = (
        Path(__file__).parent.parent.parent / "tests" / "resources" / "data"
    )
    training_ds = Dataset.from_csv(paths=str(resources_path / "dataset_source.csv"))

    # Keeping just 'label' and text 'category'
    training_ds = training_ds.map(
        lambda x: {"label": x["job"], "text": x["education"] + " " + x["marital"]},
    )

    return training_ds


def test_load_pipeline_with_custom_head(training_dataset, tmp_path):
    """Testing a model training inserting a class as custom heard"""

    # Pipeline configuration dict with custom head
    config = PipelineConfiguration(
        "test-pipeline",
        head=TaskHeadConfiguration(
            type=MyCustomHead,
            labels=[
                "blue-collar",
                "technician",
                "management",
                "services",
                "retired",
                "admin.",
            ],
        ),
        features=FeaturesConfiguration(),
    )

    # Asserting that pipeline.head is an instance of MyCustomHead
    pipeline = Pipeline.from_config(config)
    assert isinstance(pipeline.head, MyCustomHead)

    # Saving the pipeline to output
    output = tmp_path / "pipeline"
    pipeline.save(output)

    # Loading model from output
    trained_pl = Pipeline.from_pretrained(os.path.join(str(output), "model.tar.gz"))
    trained_pl.predict("Oh yeah")

    # Asserting that the pipeline head is recognized as `MyCustomHead` instance after loading from a model.tar.gz
    assert isinstance(trained_pl.head, MyCustomHead)
