from pathlib import Path

import pytest

from biome.text import Pipeline, Dataset
from biome.text.configuration import (
    TrainerConfiguration,
    FindLRConfiguration,
    VocabularyConfiguration,
)



@pytest.fixture
def train_data_source() -> Dataset:
    """Creates the training dataset"""
    resources_path = Path(__file__).parent.parent / "resources" / "data"
    training_ds = Dataset.from_csv(paths=str(resources_path / "business.cat.2k.train.csv"))

    return training_ds


@pytest.fixture
def pipeline_dict() -> dict:
    """Pipeline config dict. Updating the labels is needed"""

    pipeline_dict = {
        "name": "german_business_names",
        "features": {"word": {"embedding_dim": 16,},},
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
    vocab_config = VocabularyConfiguration(sources=[train_data_source])
    pl.create_vocabulary(vocab_config)

    # Test prediction
    prev_prediction = pl.predict("test")

    # Find_lr method
    learning_rates, losses = pl.find_lr(
        trainer_config=trainer_config,
        find_lr_config=find_lr_config,
        training_data=train_data_source,
    )

    assert len(learning_rates) == len(losses) == 12
    assert all(
        a == pytest.approx(b, rel=0.0001)
        for a, b in zip(prev_prediction["logits"], pl.predict("test")["logits"])
    )
