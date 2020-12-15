from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from biome.text import Dataset
from biome.text import Pipeline
from biome.text.configuration import FindLRConfiguration
from biome.text.configuration import TrainerConfiguration
from biome.text.configuration import VocabularyConfiguration


@pytest.fixture
def train_data_source() -> Dataset:
    """Creates the training dataset"""
    resources_path = Path(__file__).parent.parent / "resources" / "data"
    training_ds = Dataset.from_csv(
        paths=str(resources_path / "business.cat.2k.train.csv")
    )

    return training_ds


@pytest.fixture
def pipeline_dict() -> dict:
    """Pipeline config dict. Updating the labels is needed"""

    pipeline_dict = {
        "name": "german_business_names",
        "features": {
            "word": {
                "embedding_dim": 16,
            },
        },
        "head": {
            "type": "TextClassification",
            "labels": [
                "Unternehmensberatungen",
                "Friseure",
                "Tiefbau",
                "Dienstleistungen",
                "Gebrauchtwagen",
                "Restaurants",
                "Architekturbüros",
                "Elektriker",
                "Vereine",
                "Versicherungsvermittler",
                "Sanitärinstallationen",
                "Edv",
                "Maler",
                "Physiotherapie",
                "Werbeagenturen",
                "Apotheken",
                "Vermittlungen",
                "Hotels",
                "Autowerkstätten",
                "Elektrotechnik",
                "Allgemeinärzte",
                "Handelsvermittler Und -vertreter",
            ],
            "pooler": {
                "type": "gru",
                "num_layers": 1,
                "hidden_size": 16,
                "bidirectional": True,
            },
            "feedforward": {
                "num_layers": 1,
                "hidden_dims": [16],
                "activations": ["relu"],
                "dropout": [0.1],
            },
        },
    }

    return pipeline_dict


@pytest.fixture
def trainer_config() -> TrainerConfiguration:
    """Returns the trainer configuration"""
    return TrainerConfiguration(
        batch_size=64, num_epochs=5, optimizer={"type": "adam", "lr": 0.01}
    )


@pytest.fixture
def find_lr_config() -> FindLRConfiguration:
    """Returns FindLRConfiguration function"""
    return FindLRConfiguration(num_batches=11)


def test_find_lr(train_data_source, pipeline_dict, trainer_config, find_lr_config):
    """Asserting that all metrics of the find_lr method match in a controlled enviroment"""

    # Creation of pipeline and vocabulary
    pl = Pipeline.from_config(pipeline_dict)

    # Test prediction
    prev_prediction = pl.predict("test")

    # Find_lr method
    learning_rates, losses = pl.find_lr(
        trainer_config=trainer_config,
        find_lr_config=find_lr_config,
        training_data=train_data_source,
    )

    assert len(learning_rates) == len(losses) == 12
    assert_allclose(
        prev_prediction["probabilities"], pl.predict("test")["probabilities"], rtol=1e-6
    )
