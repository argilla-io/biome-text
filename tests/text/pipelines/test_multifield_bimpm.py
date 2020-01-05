import os
import pytest

from biome.text.pipelines.sequence_classifier import SequenceClassifier
from tests.test_context import TEST_RESOURCES
import pandas as pd
import yaml

BASE_CONFIG_PATH = os.path.join(TEST_RESOURCES, "resources/models/sequence_classifier")


@pytest.fixture
def training_data_yaml(tmpdir):
    data_file = tmpdir.join("sentences.csv")
    df = pd.DataFrame(
        {
            "tokens": ["Two simple sentences. Split by a dot.", "One simple sentence."],
            "label": [1, 0],
        }
    )
    df.to_csv(data_file, index=False)

    yaml_file = tmpdir.join("training.yml")
    yaml_dict = {
        "source": str(data_file),
        "mapping": {"tokens": "tokens", "label": "label"},
    }
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)
    return str(yaml_file)


@pytest.fixture
def pipeline_yaml(tmpdir):
    yaml_dict = {
        "pipeline": {
            "token_indexers": {
                "tokens": {
                    "type": "single_id"
                }
            },
            "segment_sentences": True,
            "as_text_field": True,
        },
        "architecture": {
            "text_field_embedder": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 2,
                }
            },
            "seq2vec_encoder": {
                "type": "gru",
                "input_size": 2,
                "hidden_size": 2,
                "bidirectional": False,
            },
            "multifield_seq2vec_encoder": {
                "type": "gru",
                "input_size": 2,
                "hidden_size": 2,
                "bidirectional": False,
            }
        }
    }

    yaml_file = tmpdir.join("pipeline.yml")
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


@pytest.fixture
def trainer_yaml(tmpdir):
    yaml_dict = {
        "iterator": {
            "batch_size": 2,
            "cache_instances": True,
            "max_instances_in_memory": 2,
            "sorting_keys": [["tokens", "num_fields"]],
            "type": "bucket",
        },
        "trainer": {
            "type": "default",
            "cuda_device": -1,
            "num_serialized_models_to_keep": 1,
            "num_epochs": 1,
            "optimizer": {
                "type": "adam",
                "amsgrad": True,
                "lr": 0.01,
            }
        }
    }

    yaml_file = tmpdir.join("trainer.yml")
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


def test_segment_sentences(training_data_yaml, pipeline_yaml, trainer_yaml, tmpdir, tmpdir_factory):
    pipeline = SequenceClassifier.from_config(pipeline_yaml)

    pipeline.learn(
        trainer=trainer_yaml,
        train=training_data_yaml,
        validation="",
        output=str(tmpdir.join("output")),
    )


