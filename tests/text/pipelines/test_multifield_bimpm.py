import pytest

from biome.text.pipelines.multifield_bimpm import MultifieldBiMpmPipeline
import pandas as pd
import yaml


@pytest.fixture
def training_data_yaml(tmpdir):
    data_file = tmpdir.join("multifield.csv")
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

    yaml_file = tmpdir.join("training.yml")
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


@pytest.fixture
def pipeline_yaml(tmpdir):
    yaml_dict = {
        "type": "multifield_bimpm",
        "pipeline": {
            "token_indexers": {"tokens": {"type": "single_id"}},
            "as_text_field": False,
        },
        "architecture": {
            "text_field_embedder": {
                "tokens": {
                    "type": "embedding",
                    "padding_index": 0,
                    "embedding_dim": 10,
                }
            },
            "matcher_word": {
                "is_forward": True,
                "hidden_dim": 10,
                "num_perspectives": 10,
                "with_full_match": False,
            },
            "encoder": {
                "type": "lstm",
                "bidirectional": True,
                "input_size": 10,
                "hidden_size": 200,
                "num_layers": 1,
            },
            "matcher_forward": {
                "is_forward": True,
                "hidden_dim": 200,
                "num_perspectives": 10,
            },
            "matcher_backward": {
                "is_forward": False,
                "hidden_dim": 200,
                "num_perspectives": 10,
            },
            "aggregator": {
                "type": "gru",
                "bidirectional": True,
                "input_size": 154,
                "hidden_size": 100,
            },
            "classifier_feedforward": {
                "input_dim": 400,
                "num_layers": 1,
                "hidden_dims": [200],
                "activations": ["relu"],
                "dropout": [0.0],
            },
            "initializer": [
                [".*linear_layers.*weight", {"type": "xavier_normal"}],
                [".*linear_layers.*bias", {"type": "constant", "val": 0}],
                [".*weight_ih.*", {"type": "xavier_normal"}],
                [".*weight_hh.*", {"type": "orthogonal"}],
                [".*bias.*", {"type": "constant", "val": 0}],
                [".*matcher.*match_weights.*", {"type": "kaiming_normal"}],
            ],
        },
    }

    yaml_file = tmpdir.join("pipeline.yml")
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


@pytest.fixture
def trainer_yaml(tmpdir):
    yaml_dict = {
        "iterator": {
            "batch_size": 3,
            "sorting_keys": [["record1", "num_fields"]],
            "type": "bucket",
        },
        "trainer": {
            "type": "default",
            "cuda_device": -1,
            "num_serialized_models_to_keep": 1,
            "num_epochs": 1,
            "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.01,},
        },
    }

    yaml_file = tmpdir.join("trainer.yml")
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


def test_multifield_bimpm_learn(
    training_data_yaml, pipeline_yaml, trainer_yaml, tmpdir, tmpdir_factory
):
    pipeline = MultifieldBiMpmPipeline.from_config(pipeline_yaml)

    pipeline.learn(
        trainer=trainer_yaml,
        train=training_data_yaml,
        validation="",
        output=str(tmpdir.join("output")),
    )
