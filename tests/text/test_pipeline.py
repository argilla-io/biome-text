import pytest

from biome.text import Pipeline, PipelineConfiguration
from biome.text.configuration import FeaturesConfiguration
from biome.text.features import CharFeatures
from biome.text.modules.heads import TaskHeadConfiguration, TextClassification


@pytest.mark.skip()
def test_pipeline_prediction_with_empty_inputs():

    pipeline = Pipeline.from_config(
        PipelineConfiguration(
            name="test",
            head=TaskHeadConfiguration(type=TextClassification, labels=["a", "b"]),
            features=FeaturesConfiguration(
                char=CharFeatures(
                    embedding_dim=10,
                    encoder={
                        "type": "cnn",
                        "num_filters": 50,
                        "ngram_filter_sizes": [4],
                    },
                )
            ),
        )
    )

    assert pipeline.predict("a") is None
