from typing import Dict

import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration
from biome.text.modules.heads.task_prediction import Attribution


@pytest.fixture
def training_dataset() -> Dataset:
    """Creating the dataframe."""
    data = {
        "record1": [
            {"@first_name": "Hans", "@last_name": "Peter"},
            {"@first_name": "Heinrich", "@last_name": "Meier"},
            {"@first_name": "Hans", "@last_name": "Peter"},
        ],
        "record2": [
            {"@first_name": "Hans", "@last_name": "Petre"},
            {"@first_name": "Heinz", "@last_name": "Meier"},
            {"@first_name": "Hansel", "@last_name": "Peter"},
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
def trainer_config() -> TrainerConfiguration:
    return TrainerConfiguration(
        max_epochs=1,
        optimizer={"type": "adam", "amsgrad": True, "lr": 0.002},
        gpus=0,
    )


def test_train(pipeline_dict, training_dataset, trainer_config, tmp_path):
    """Testing the correct working of prediction, vocab creating and training"""
    pipeline = Pipeline.from_config(pipeline_dict)
    pipeline.predict(record1={"first_name": "Hans"}, record2={"first_name": "Hansel"})

    trainer = Trainer(
        pipeline=pipeline,
        train_dataset=training_dataset,
        valid_dataset=training_dataset,
        trainer_config=trainer_config,
    )
    trainer.fit(tmp_path / "record_bimpm_experiment")


def test_attributions(pipeline_dict, training_dataset):
    pipeline = Pipeline.from_config(pipeline_dict)
    instance = pipeline.head.featurize(
        training_dataset["record1"][0], training_dataset["record2"][0]
    )
    pipeline.model.eval()
    forward_output = pipeline.model.forward_on_instances([instance])

    attributions = pipeline.head._compute_attributions(forward_output[0], instance)

    assert all([isinstance(attribution, Attribution) for attribution in attributions])
    assert len(attributions) == 4
    assert all([isinstance(attr.attribution, float) for attr in attributions])
    assert all([attributions[i].field == "record1" for i in [0, 1]])
    assert all([attributions[i].field == "record2" for i in [2, 3]])
    assert attributions[1].start == 0 and attributions[1].end == 16

    assert attributions[0].text == "@first_name Hans"
    assert attributions[3].text == "@last_name Petre"

    # Raise error when records with different number of record fields
    instance = pipeline.head.featurize(
        record1={"first_name": "Hans", "last_name": "Zimmermann"},
        record2={"first_name": "Hansel"},
    )
    forward_output = pipeline._model.forward_on_instances([instance])

    with pytest.raises(RuntimeError):
        pipeline.head._compute_attributions(forward_output[0], instance)
