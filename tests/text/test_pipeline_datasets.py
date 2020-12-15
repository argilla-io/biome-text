import logging
import os

import pytest
import torch

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import PipelineConfiguration
from biome.text import TrainerConfiguration
from biome.text.modules.heads import TextClassificationConfiguration
from tests.text.test_pipeline_model import TestHead


@pytest.fixture
def dataset(tmp_path) -> Dataset:
    data = {
        "text": ["A common text", "This is why you get", "Seriosly?, I'm not sure"],
        "label": ["one", "zero", "zero"],
    }
    ds = Dataset.from_dict(data)

    # we save and load it here to be able to lazily read from it
    ds_path = tmp_path / "test_pipeline_datasets" / "dataset"
    ds.save_to_disk(str(ds_path))

    return Dataset.load_from_disk(str(ds_path))


@pytest.fixture
def pipeline() -> Pipeline:
    config = PipelineConfiguration(
        name="test-classifier",
        head=TextClassificationConfiguration(labels=["one", "zero"]),
    )
    return Pipeline.from_config(config)


def test_training_with_data_bucketing(
    pipeline: Pipeline, dataset: Dataset, tmp_path: str
):
    configuration = TrainerConfiguration(
        data_bucketing=True, batch_size=2, num_epochs=5
    )

    pipeline.copy().train(
        output=os.path.join(tmp_path, "output"),
        trainer=configuration,
        training=dataset,
        validation=dataset,
        lazy=False,
    )

    pipeline.copy().train(
        output=os.path.join(tmp_path, "output"),
        trainer=configuration,
        training=dataset,
        validation=dataset,
        lazy=True,
    )


def test_training_from_pretrained_with_head_replace(
    pipeline: Pipeline, dataset: Dataset, tmp_path: str
):
    configuration = TrainerConfiguration(
        data_bucketing=True,
        batch_size=2,
        num_epochs=5,
        cuda_device=-1,
    )
    output_dir = os.path.join(tmp_path, "output")
    pipeline.train(
        output=output_dir, trainer=configuration, training=dataset, quiet=True
    )

    pipeline.set_head(TestHead)
    pipeline.config.tokenizer_config.max_nr_of_sentences = 3
    copied = pipeline.copy()
    assert isinstance(copied.head, TestHead)
    assert copied.num_parameters == pipeline.num_parameters
    assert copied.num_trainable_parameters == pipeline.num_trainable_parameters
    copied_model_state = copied._model.state_dict()
    original_model_state = pipeline._model.state_dict()
    for key, value in copied_model_state.items():
        if "backbone" in key:
            assert torch.all(torch.eq(value, original_model_state[key]))
    assert copied.backbone.featurizer.tokenizer.config.max_nr_of_sentences == 3


def test_training_with_logging(pipeline: Pipeline, dataset: Dataset, tmp_path: str):
    configuration = TrainerConfiguration(
        data_bucketing=True, batch_size=2, num_epochs=5
    )
    output_dir = os.path.join(tmp_path, "output")
    pipeline.train(
        output=output_dir, trainer=configuration, training=dataset, quiet=True
    )

    assert os.path.exists(os.path.join(output_dir, "train.log"))
    with open(os.path.join(output_dir, "train.log")) as train_log:
        for line in train_log.readlines()[3:]:
            assert "allennlp" in line

    assert logging.getLogger("allennlp").level == logging.ERROR
    assert logging.getLogger("biome").level == logging.INFO
