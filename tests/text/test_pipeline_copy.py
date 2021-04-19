import pytest
from numpy.testing import assert_allclose

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration


@pytest.fixture
def pipeline():
    return Pipeline.from_config(
        {
            "name": "test_pipeline_copy",
            "head": {
                "type": "TextClassification",
                "labels": ["a", "b"],
            },
        }
    )


@pytest.fixture
def dataset():
    return Dataset.from_dict(
        {
            "text": ["this is", "a test"],
            "label": ["a", "b"],
        }
    )


def test_copy(pipeline):
    prediction = pipeline.predict("check this")
    pipeline_copy = pipeline.copy()
    prediction_copy = pipeline_copy.predict("check this")

    assert_allclose(prediction["probabilities"], prediction_copy["probabilities"])


def test_train_from_pretrained(pipeline, dataset, tmp_path):
    output_path = tmp_path / "test_train_from_pretrained_output"
    trainer_config = TrainerConfiguration(max_epochs=1, batch_size=2, gpus=0)
    trainer = Trainer(
        pipeline=pipeline, train_dataset=dataset, trainer_config=trainer_config
    )
    trainer.fit(output_path)

    prediction = pipeline.predict("a test")
    pipeline_loaded = Pipeline.from_pretrained(output_path / "model.tar.gz")
    prediction_loaded = pipeline_loaded.predict("a test")

    assert_allclose(prediction["probabilities"], prediction_loaded["probabilities"])
