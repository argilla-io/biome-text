import os
from tempfile import mkdtemp

from biome.text import Pipeline, PipelineConfiguration
from biome.text.configuration import (
    FeaturesConfiguration,
    TrainerConfiguration,
    VocabularyConfiguration,
)
from biome.text.data import DataSource
from biome.text.modules.heads import TaskHeadConfiguration, TextClassification
from tests.test_context import TEST_RESOURCES


class MyCustomHead(TextClassification):
    """Just a head renaming the original TextClassification head"""

    pass


def test_load_pipeline_with_custom_head():
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

    pipeline = Pipeline.from_config(config)
    assert isinstance(pipeline.head, MyCustomHead)

    train = DataSource(
        source=os.path.join(TEST_RESOURCES, "resources/data/dataset_source.csv"),
        mapping={"label": "job", "text": ["education", "marital"]},
    )
    output = mkdtemp()
    pipeline.create_vocabulary(VocabularyConfiguration(sources=[train]))
    pipeline.train(output=output, training=train)

    trained_pl = Pipeline.from_pretrained(os.path.join(output, "model.tar.gz"))
    trained_pl.predict("Oh yeah")
    assert isinstance(trained_pl.head, MyCustomHead)
