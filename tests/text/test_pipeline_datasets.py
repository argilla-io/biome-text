import logging
import os

import pandas as pd
import pytest

# fmt: off
import torch

from biome.text import (
    Pipeline,
    Dataset,
    PipelineConfiguration,
    TrainerConfiguration,
    VocabularyConfiguration,
)
from biome.text.modules.heads import TextClassificationConfiguration

from allennlp.data import AllennlpDataset, Instance, AllennlpLazyDataset
# fmt: on
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
def pipeline_test() -> Pipeline:
    config = PipelineConfiguration(
        name="test-classifier",
        head=TextClassificationConfiguration(labels=["one", "zero"]),
    )
    return Pipeline.from_config(config)


def test_training_with_data_bucketing(
    pipeline_test: Pipeline, dataset: Dataset, tmp_path: str
):
    lazy_instances = dataset.to_instances(pipeline_test)
    in_memory_instances = dataset.to_instances(pipeline_test, lazy=False)

    pipeline_test.create_vocabulary(VocabularyConfiguration(sources=[lazy_instances]))

    configuration = TrainerConfiguration(
        data_bucketing=True, batch_size=2, num_epochs=5
    )
    pipeline_test.train(
        output=os.path.join(tmp_path, "output"),
        trainer=configuration,
        training=lazy_instances,
        validation=in_memory_instances,
    )

    pipeline_test.train(
        output=os.path.join(tmp_path, "output"),
        trainer=configuration,
        training=in_memory_instances,
        validation=lazy_instances,
    )


def test_training_from_pretrained_with_head_replace(
    pipeline_test: Pipeline, dataset: Dataset, tmp_path: str
):
    training = dataset.to_instances(pipeline_test)
    pipeline_test.create_vocabulary(VocabularyConfiguration(sources=[training]))
    configuration = TrainerConfiguration(
        data_bucketing=True, batch_size=2, num_epochs=5
    )
    output_dir = os.path.join(tmp_path, "output")
    results = pipeline_test.train(
        output=output_dir, trainer=configuration, training=training, quiet=True
    )

    trained = Pipeline.from_pretrained(results.model_path)
    trained.set_head(TestHead)
    trained.config.tokenizer_config.max_nr_of_sentences = 3
    copied = trained.copy()
    assert isinstance(copied.head, TestHead)
    assert copied.num_parameters == trained.num_parameters
    assert copied.num_trainable_parameters == trained.num_trainable_parameters
    copied_model_state = copied._model.state_dict()
    original_model_state = trained._model.state_dict()
    for key, value in copied_model_state.items():
        if "backbone" in key:
            assert torch.all(torch.eq(value, original_model_state[key]))
    assert copied.backbone.featurizer.tokenizer.config.max_nr_of_sentences == 3


def test_training_with_logging(
    pipeline_test: Pipeline, dataset: Dataset, tmp_path: str
):
    training = dataset.to_instances(pipeline_test)
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
