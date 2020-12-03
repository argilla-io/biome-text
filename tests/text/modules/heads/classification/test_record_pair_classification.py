from typing import Dict

import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration


@pytest.fixture
def training_dataset() -> Dataset:
    """Creating the dataframe."""
    data = {
        "record1": [
            {"@fist_name": "Hans", "@last_name": "Peter"},
            {"@fist_name": "Heinrich", "@last_name": "Meier"},
            {"@fist_name": "Hans", "@last_name": "Peter"},
        ],
        "record2": [
            {"@fist_name": "Hans", "@last_name": "Petre"},
            {"@fist_name": "Heinz", "@last_name": "Meier"},
            {"@fist_name": "Hansel", "@last_name": "Peter"},
        ],
        "label": ["duplicate", "not_duplicate", "duplicate"],
    }

    return Dataset.from_dict(data)


@pytest.fixture
def pipeline_dict() -> Dict:
    """Creating the pipeline dictionary"""

    pipeline_dict = {
        "name": "biome-bimpm",
        "tokenizer": {"text_cleaning": {"rules": ["strip_spaces"]}},
        "features": {
            # "tokens": {
            #     "indexer": {"type": "single_id", "lowercase_tokens": True},
            #     "embedder": {
            #         "type": "embedding",
            #         "embedding_dim": 32,
            #         "padding_index": 0,
            #     },
            # },
            "word": {
                "embedding_dim": 2,
                "lowercase_tokens": True,
                "trainable": True,
                "embedder": {"padding_index": 0},
            },
            # "token_characters": {
            #     "indexer": {
            #         "type": "characters",
            #         "character_tokenizer": {"lowercase_characters": True},
            #     },
            #     "embedder": {
            #         "type": "character_encoding",
            #         "embedding": {"embedding_dim": 32, "padding_index": 0},
            #         "encoder": {
            #             "type": "gru",
            #             "input_size": 32,
            #             "hidden_size": 112,
            #             "num_layers": 1,
            #             "bidirectional": True,
            #         },
            #         "dropout": 0.2,
            #     },
            # },
            "char": {
                "embedding_dim": 2,
                "dropout": 0.1,
                "encoder": {
                    "type": "gru",
                    "hidden_size": 2,
                    "num_layers": 1,
                    "bidirectional": True,
                },
                "indexer": {"character_tokenizer": {"lowercase_characters": True}},
                "embedder": {"embedding": {"padding_index": 0}},
            },
        },
        "head": {
            "type": "RecordPairClassification",
            "labels": ["duplicate", "not_duplicate"],
            "dropout": 0.1,
            "field_encoder": {
                "type": "gru",
                "bidirectional": False,
                # "input_size": 4,
                "hidden_size": 4,
                "num_layers": 1,
            },
            "record_encoder": {
                "type": "gru",
                "bidirectional": True,
                # "input_size": 4,
                "hidden_size": 2,
                "num_layers": 1,
            },
            "matcher_forward": {
                "is_forward": True,
                # "hidden_dim": 2,
                "num_perspectives": 10,
                "with_full_match": False,
            },
            "matcher_backward": {
                "is_forward": False,
                # "hidden_dim": 2,
                "num_perspectives": 10,
            },
            "aggregator": {
                "type": "gru",
                "bidirectional": True,
                # "input_size": ??,
                "hidden_size": 2,
                "num_layers": 1,
            },
            "classifier_feedforward": {
                # "input_dim": 8,
                "num_layers": 1,
                "hidden_dims": [4],
                "activations": ["relu"],
                "dropout": [0.1],
            },
            "initializer": {
                "regexes": [
                    ["_output_layer.weight", {"type": "xavier_normal"}],
                    ["_output_layer.bias", {"type": "constant", "val": 0}],
                    [".*linear_layers.*weight", {"type": "xavier_normal"}],
                    [".*linear_layers.*bias", {"type": "constant", "val": 0}],
                    [".*weight_ih.*", {"type": "xavier_normal"}],
                    [".*weight_hh.*", {"type": "orthogonal"}],
                    [".*bias.*", {"type": "constant", "val": 0}],
                    [".*matcher.*match_weights.*", {"type": "kaiming_normal"}],
                ]
            },
        },
    }

    return pipeline_dict


@pytest.fixture
def trainer_dict() -> Dict:
    """Creating the trainer dictionary"""

    trainer_dict = {
        "num_epochs": 1,
        "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.002},
    }

    return trainer_dict


def test_explain(pipeline_dict):
    """Checking expected parammeters of two rows of a dataset"""

    pipeline = Pipeline.from_config(pipeline_dict)
    explain = pipeline.explain(
        record1={"first_name": "Hans"},
        record2={"first_name": "Hansel"},
    )
    # Records 1 and 2 must have the same dictionary length, and must be equal to the string assigned to them
    assert len(explain["explain"]["record1"]) == len(explain["explain"]["record2"])
    assert explain["explain"]["record1"][0]["token"] == "first_name Hans"
    assert explain["explain"]["record2"][0]["token"] == "first_name Hansel"

    # Checking .explain method
    with pytest.raises(RuntimeError):
        pipeline.explain(
            record1={"first_name": "Hans", "last_name": "Zimmermann"},
            record2={"first_name": "Hansel"},
        )


def test_train(pipeline_dict, training_dataset, trainer_dict, tmp_path):
    """Testing the correct working of prediction, vocab creating and training"""
    pipeline = Pipeline.from_config(pipeline_dict)
    pipeline.predict(record1={"first_name": "Hans"}, record2={"first_name": "Hansel"})

    pipeline.train(
        output=str(tmp_path / "record_bimpm_experiment"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_dataset,
        validation=training_dataset,
    )
