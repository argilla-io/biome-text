from typing import Dict

import pandas as pd
import pytest
import yaml
from biome.text import TrainerConfiguration
from biome.text import Pipeline


@pytest.fixture
def path_to_training_data_yaml(tmp_path) -> str:
    data_file = tmp_path / "multifield.csv"
    df = pd.DataFrame(
        {
            "record1_1": ["record1_1", "record1_1 test", "record1_1 test this"],
            "record1_2": ["record1_2", "record1_2 test", "record1_2 test this"],
            "record1_3": ["record1_3", "record1_3 test", "record1_3 test this"],
            "record2_1": ["record2_1", "record2_1 test", "record2_1 test this"],
            "record2_2": ["record2_2", "record2_2 test", "record2_2 test this"],
            "label": [1, 0, 1],
        }
    )
    df.to_csv(data_file, index=False)

    yaml_file = tmp_path / "training.yml"
    yaml_dict = {
        "source": str(data_file),
        "mapping": {
            "record1": ["record1_1", "record1_2", "record1_3"],
            "record2": ["record2_1", "record2_2"],
            "label": "label",
        },
    }
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


@pytest.fixture(params=["singlefield", "multifield"])
def path_to_pipeline_yaml(tmp_path, request) -> str:
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
            "words": {
                "embedding_dim": 32,
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
            "chars": {
                "embedding_dim": 32,
                "dropout": 0.2,
                "encoder": {
                    "type": "gru",
                    "hidden_size": 112,
                    "num_layers": 1,
                    "bidirectional": True,
                },
                "indexer": {"character_tokenizer": {"lowercase_characters": True}},
                "embedder": {"embedding": {"padding_index": 0}},
            },
        },
        "head": {
            "type": "BiMpm",
            "labels": ["1", "0"],
            "multifield": request.param == "multifield",
            "dropout": 0.2,
            "encoder": {
                "type": "lstm",
                "bidirectional": False,
                # "input_size": 256,
                "hidden_size": 64,
                "num_layers": 1,
            },
            "matcher_word": {
                "is_forward": True,
                # "hidden_dim": 256,
                "num_perspectives": 10,
                "with_full_match": False,
            },
            "matcher_forward": {
                "is_forward": True,
                # "hidden_dim": 64,
                "num_perspectives": 21,
            },
            "encoder2": {
                "type": "lstm",
                "bidirectional": False,
                # "input_size": 64,
                "hidden_size": 32,
                "num_layers": 1,
            },
            "matcher2_forward": {
                "is_forward": True,
                # "hidden_dim": 32,
                "num_perspectives": 21,
            },
            "aggregator": {
                "type": "lstm",
                "bidirectional": True,
                # "input_size": 264,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "classifier_feedforward": {
                # "input_dim": 128,
                "num_layers": 1,
                "hidden_dims": [64],
                "activations": ["relu"],
                "dropout": [0.2],
            },
            "initializer": [
                ["output_layer.weight", {"type": "xavier_normal"}],
                ["output_layer.bias", {"type": "constant", "val": 0}],
                [".*linear_layers.*weight", {"type": "xavier_normal"}],
                [".*linear_layers.*bias", {"type": "constant", "val": 0}],
                [".*weight_ih.*", {"type": "xavier_normal"}],
                [".*weight_hh.*", {"type": "orthogonal"}],
                [".*bias.*", {"type": "constant", "val": 0}],
                [".*matcher.*match_weights.*", {"type": "kaiming_normal"}],
            ],
        },
    }
    pipeline_yaml = tmp_path / "pipeline.yml"
    with pipeline_yaml.open("w") as f:
        yaml.safe_dump(pipeline_dict, f)

    return str(pipeline_yaml)


@pytest.fixture
def trainer_dict() -> Dict:
    trainer_dict = {
        "num_epochs": 1,
        "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.01},
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.2,
            "mode": "min",
            "patience": 1,
        },
        "validation_metric": "-loss",
        "patience": 5,
    }

    return trainer_dict


def test_bimpm_train(
    path_to_pipeline_yaml, trainer_dict, path_to_training_data_yaml,
):
    pipeline = Pipeline.from_file(path_to_pipeline_yaml)
    pipeline.predict(record1="The one", record2="The other")

    pipeline.train(
        output="experiment",
        trainer=TrainerConfiguration(**trainer_dict),
        training=path_to_training_data_yaml,
        validation=path_to_training_data_yaml,
        restore=False,
    )
