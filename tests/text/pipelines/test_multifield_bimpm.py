import pytest

from biome.text.api_new.pipeline import Pipeline
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


@pytest.fixture(params=["singlefield", "multifield"])
def pipeline_yaml(tmpdir, request):
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
            # Missing: padding_index
            "words": {
                "embedding_dim": 32,
                "lowercase_tokens": True,
                "trainable": True,
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
            # Missing: lowercase_characters, padding_index
            "chars": {
                "embedding_dim": 32,
                "dropout": 0.2,
                "encoder": {
                    "type": "gru",
                    "hidden_size": 112,
                    "num_layers": 1,
                    "bidirectional": True,
                }
            }
        },
        "head": {
            "type": "BiMpm",
            "labels": ["1", "0"],
            "matcher_word": {
                "is_forward": True,
                "hidden_dim": 256,
                "num_perspectives": 10,
                "with_full_match": False,
            },
            "encoder": {
                "type": "lstm",
                "bidirectional": False,
                "input_size": 256,
                "hidden_size": 64,
                "num_layers": 1,
            },
            "matcher_forward": {
                "is_forward": True,
                "hidden_dim": 64,
                "num_perspectives": 21,
            },
            "encoder2": {
                "type": "lstm",
                "bidirectional": False,
                "input_size": 64,
                "hidden_size": 32,
                "num_layers": 1,
            },
            "matcher2_forward": {
                "is_forward": True,
                "hidden_dim": 32,
                "num_perspectives": 21,
            },
            "aggregator": {
                "type": "lstm",
                "bidirectional": True,
                "input_size": 264,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "classifier_feedforward": {
                "input_dim": 128,
                "num_layers": 1,
                "hidden_dims": [64],
                "activations": ["relu"],
                "dropout": [0.2],
            },
            "multifield": request.param == "multifield",
            "dropout": 0.2,
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
    pipeline_yaml = tmpdir.join("pipeline.yml")
    with pipeline_yaml.open("w") as f:
        yaml.safe_dump(pipeline_dict, f)

    return str(pipeline_yaml)


@pytest.fixture
def trainer_yaml(tmpdir):
    trainer_dict = {
        # "iterator": {
        #     "batch_size": 3,
        #     "sorting_keys": [
        #         [
        #             "record1",
        #             "num_fields" if request.param == "multifield" else "num_tokens",
        #         ]
        #     ],
        #     "type": "bucket",
        # },
        # Missing: batch_size; max_instances_in_memory;
        "trainer": {
            "type": "default",
            "cuda_device": -1,
            "num_serialized_models_to_keep": 1,
            "should_log_learning_rate": True,
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
        },
    }
    trainer_yaml = tmpdir.join("trainer.yml")
    with trainer_yaml.open("w") as f:
        yaml.safe_dump(trainer_dict, f)

    return str(trainer_yaml)


def test_multifield_bimpm_learn(
    pipeline_yaml, trainer_yaml, training_data_yaml,
):
    pipeline = Pipeline.from_file(pipeline_yaml)
    print(pipeline.predict(record1="The one", record2="The other"))

    pipeline.train(
        output="experiment",
        trainer=trainer_yaml,
        training=training_data_yaml,
        validation=training_data_yaml,
    )
