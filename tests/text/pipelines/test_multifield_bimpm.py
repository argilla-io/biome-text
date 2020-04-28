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


@pytest.fixture
def trainer_yaml(tmpdir):
    yaml_dict = {
        "iterator": {
            "batch_size": 3,
            "sorting_keys": [["record1", "num_tokens"]],
            # "sorting_keys": [["record1", "num_fields"]],
            "type": "bucket",
        },
        "trainer": {
            "type": "default",
            "cuda_device": -1,
            "num_serialized_models_to_keep": 1,
            "should_log_learning_rate": True,
            "num_epochs": 1,
            "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.01},
            "learning_rate_scheduler": {"type": "reduce_on_plateau", "factor": 0.2, "mode": "min", "patience": 1},
            "validation_metric": "-loss",
            "patience": 5,
        },
    }

    yaml_file = tmpdir.join("trainer.yml")
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


@pytest.fixture
def pipeline_yaml():
    return "./resources/bimpm.yaml"


def test_multifield_bimpm_learn(
    training_data_yaml, pipeline_yaml, trainer_yaml, tmpdir, tmpdir_factory
):
    pipeline = Pipeline.from_file("resources/bimpm.yaml")
    # print(pipeline.predict(record1="The one", record2="The other"))

    trained_pl = pipeline.train(
        output="experiment",
        trainer=trainer_yaml,
        training=training_data_yaml,
        validation=training_data_yaml,
    )

