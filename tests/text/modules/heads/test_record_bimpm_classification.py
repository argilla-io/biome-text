from typing import Dict

import pandas as pd
import pytest
import yaml
from biome.text import TrainerConfiguration, VocabularyConfiguration
from biome.text import Pipeline
from biome.text.data import DataSource


@pytest.fixture
def path_to_training_data_yaml(tmp_path) -> str:
    data_file = tmp_path / "record_pairs.json"
    df = pd.DataFrame(
        {
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
    )
    df.to_json(data_file, lines=True, orient="records")

    yaml_file = tmp_path / "training.yml"
    yaml_dict = {
        "source": str(data_file),
        "attributes": {"flatten": False}
        # "mapping": {
        #     "record1": "record1",
        #     "record2": "record2",
        #     "label": "label",
        # },
    }
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


@pytest.fixture
def path_to_pipeline_yaml(tmp_path) -> str:
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
            "type": "RecordBiMpm",
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
            "initializer": [
                ["_output_layer.weight", {"type": "xavier_normal"}],
                ["_output_layer.bias", {"type": "constant", "val": 0}],
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
        "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.002},
    }

    return trainer_dict


def test_record_bimpm_train(
    path_to_pipeline_yaml, trainer_dict, path_to_training_data_yaml, tmp_path
):
    pipeline = Pipeline.from_yaml(path_to_pipeline_yaml,)
    pipeline.predict(record1={"first_name": "Hans"}, record2={"first_name": "Hansel"})
    pipeline.create_vocabulary(
        VocabularyConfiguration(
            sources=[DataSource.from_yaml(path_to_training_data_yaml)]
        )
    )

    pipeline.train(
        output=str(tmp_path / "record_bimpm_experiment"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=path_to_training_data_yaml,
        validation=path_to_training_data_yaml,
    )
