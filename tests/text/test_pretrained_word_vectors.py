from pathlib import Path

import pytest
import torch
from torch.testing import assert_allclose

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration


@pytest.fixture
def pretrained_word_vectors(tmp_path) -> Path:
    file_path = tmp_path / "pretrained_word_vectors.txt"
    file_path.write_text("2 2\ntest 0.66 0.33\nthis 0.25 0.75")

    return file_path


@pytest.fixture
def dataset() -> Dataset:
    data = {"text": ["test"], "label": ["good"]}
    return Dataset.from_dict(data)


@pytest.fixture
def dataset2() -> Dataset:
    data = {"text": ["this"], "label": ["good"]}
    return Dataset.from_dict(data)


@pytest.fixture
def pipeline_config(pretrained_word_vectors) -> dict:
    config = {
        "name": "pretrained_word_vectors_test",
        "features": {
            "word": {
                "embedding_dim": 2,
                "weights_file": str(pretrained_word_vectors.absolute()),
            }
        },
        "head": {"type": "TextClassification", "labels": ["good"]},
    }
    return config


def test_create_pipeline_with_weights_file(pipeline_config, dataset, tmp_path):
    pipeline = Pipeline.from_config(pipeline_config)

    output = tmp_path / "pretrained_word_vector_output"
    trainer = Trainer(
        pipeline=pipeline,
        train_dataset=dataset,
        trainer_config=TrainerConfiguration(max_epochs=1, gpus=0),
    )
    trainer.fit(output)

    instance = pipeline.head.featurize("test")
    instance.index_fields(pipeline.vocab)

    assert_allclose(
        pipeline.backbone.embedder(instance.as_tensor_dict()["text"], 0),
        torch.tensor([[0.66, 0.33]]),
    )

    # Loading a pretrained model without the weights file should work
    Path(pipeline_config["features"]["word"]["weights_file"]).unlink()
    assert isinstance(Pipeline.from_pretrained(str(output / "model.tar.gz")), Pipeline)


def test_extending_vocab_with_weights_file(
    pipeline_config, dataset, dataset2, capsys, caplog
):
    pipeline = Pipeline.from_config(pipeline_config)
    # create vocab
    pipeline.create_vocab([dataset.to_instances(pipeline)])

    # extending the vocab with the weights file available should apply the pretrained weights
    pipeline.create_vocab([dataset2.to_instances(pipeline)])

    instance = pipeline.head.featurize("this")
    instance.index_fields(pipeline.vocab)

    assert_allclose(
        pipeline.backbone.embedder(instance.as_tensor_dict()["text"]),
        torch.tensor([[0.25, 0.75]]),
    )

    # extending the vocab with the weights file deleted should trigger a warning
    Path(pipeline_config["features"]["word"]["weights_file"]).unlink()
    ds = Dataset.from_dict({"text": ["that"], "label": ["good"]})
    pipeline.create_vocab([ds.to_instances(pipeline)])

    assert caplog.record_tuples[-1][0] == "allennlp.modules.token_embedders.embedding"
    assert caplog.record_tuples[-1][1] == 30
    assert (
        "Embedding at model_path, "
        "_head.backbone.embedder.token_embedder_word cannot locate the pretrained_file."
        in caplog.record_tuples[-1][2]
    )


def test_raise_filenotfound_error(pipeline_config, dataset):
    Path(pipeline_config["features"]["word"]["weights_file"]).unlink()
    pipeline = Pipeline.from_config(pipeline_config)

    with pytest.raises(FileNotFoundError):
        pipeline.create_vocab([dataset.to_instances(pipeline)])
