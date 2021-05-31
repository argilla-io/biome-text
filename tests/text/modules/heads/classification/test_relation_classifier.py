from typing import Dict

import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration


@pytest.fixture
def training_dataset() -> Dataset:
    """Creating the dataframe."""
    data = {
        "text": [
            "The most common audits were about waste and recycling.",
            "The company fabricates plastic chairs.",
        ],
        "entities": [
            [
                {"start": 34, "end": 39, "label": "PN", "text": "waste"},
                {"start": 16, "end": 22, "label": "QTY", "text": "audits"},
            ],
            [
                {"start": 4, "end": 11, "label": "OBJECT", "text": "company"},
                {"start": 31, "end": 37, "label": "SUBJECT", "text": "chairs"},
            ],
        ],
        "label": ["Message-Topic(e1,e2)", "Product-Producer(e2,e1)"],
    }

    return Dataset.from_dict(data)


@pytest.fixture
def pipeline_dict() -> Dict:
    """Creating the pipeline dictionary"""

    pipeline_dict = {
        "name": "biome-rele",
        "features": {
            "word": {"embedding_dim": 2},
            "char": {
                "embedding_dim": 2,
                "dropout": 0.1,
                "encoder": {
                    "type": "gru",
                    "hidden_size": 2,
                },
            },
        },
        "head": {
            "type": "RelationClassification",
            "labels": ["Message-Topic(e1,e2)", "Product-Producer(e2,e1)"],
            "entities_embedder": {"num_embeddings": 12, "embedding_dim": 50},
            "feedforward": {
                "num_layers": 1,
                "hidden_dims": [4],
                "activations": ["relu"],
                "dropout": [0.1],
            },
        },
    }

    return pipeline_dict


@pytest.fixture
def trainer_config() -> TrainerConfiguration:
    return TrainerConfiguration(
        max_epochs=1,
        optimizer={"type": "adamw", "lr": 0.002},
        gpus=0,
    )


def test_train(pipeline_dict, training_dataset, trainer_config, tmp_path):
    """Testing a classifier made from scratch"""

    pipeline = Pipeline.from_config(pipeline_dict)
    pipeline.predict(
        text="The most common audits were about waste and recycling",
        entities=[
            {"start": 34, "end": 39, "label": "OBJECT", "text": "waste"},
            {"start": 16, "end": 22, "label": "SUBJECT", "text": "audits"},
        ],
    )

    trainer = Trainer(
        pipeline=pipeline,
        train_dataset=training_dataset,
        valid_dataset=training_dataset,
        trainer_config=trainer_config,
    )
    trainer.fit(tmp_path / "relation_classifier")

    # test loading
    Pipeline.from_pretrained(tmp_path / "relation_classifier" / "model.tar.gz")
