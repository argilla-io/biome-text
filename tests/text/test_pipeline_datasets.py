import logging
import os

import pytest
import torch

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import PipelineConfiguration
from biome.text import Trainer
from biome.text import TrainerConfiguration
from biome.text.backbone import ModelBackbone
from biome.text.modules.heads import TextClassification
from biome.text.modules.heads import TextClassificationConfiguration


class TestHead(TextClassification):
    def __init__(self, backbone: ModelBackbone):
        super(TestHead, self).__init__(backbone, labels=["test", "notest"])


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


def test_training_with_data_bucketing(pipeline, dataset, tmp_path):
    trainer_config = TrainerConfiguration(
        data_bucketing=True, batch_size=2, max_epochs=5, gpus=0
    )

    trainer = Trainer(
        pipeline,
        train_dataset=dataset,
        valid_dataset=dataset,
        trainer_config=trainer_config,
        lazy=True,
    )
    trainer.fit(tmp_path / "output")

    trainer = Trainer(
        pipeline,
        train_dataset=dataset,
        valid_dataset=dataset,
        trainer_config=trainer_config,
        lazy=False,
    )
    trainer.fit(tmp_path / "output", exist_ok=True)


def test_training_from_pretrained_with_head_replace(pipeline, dataset, tmp_path):
    trainer_config = TrainerConfiguration(
        data_bucketing=True,
        batch_size=2,
        max_epochs=5,
        gpus=0,
    )

    trainer = Trainer(pipeline, train_dataset=dataset, trainer_config=trainer_config)
    trainer.fit(tmp_path / "output")

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
