from typing import Tuple

import pytest
from pytorch_lightning import seed_everything

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import VocabularyConfiguration
from biome.text.configuration import CharFeatures
from biome.text.configuration import TrainerConfiguration
from biome.text.configuration import WordFeatures


@pytest.fixture
def train_valid_dataset(resources_data_path) -> Tuple[Dataset, Dataset]:
    """Returns both training and validation datasets"""

    training_ds = Dataset.from_csv(
        paths=str(resources_data_path / "business.cat.2k.train.csv")
    )
    validation_ds = Dataset.from_csv(
        paths=str(resources_data_path / "business.cat.2k.valid.csv")
    )

    return training_ds, validation_ds


@pytest.fixture
def pipeline_dict() -> dict:
    return {
        "name": "german_business_names",
        "features": {
            "word": {"embedding_dim": 16, "lowercase_tokens": True},
            "char": {
                "embedding_dim": 16,
                "encoder": {
                    "type": "gru",
                    "num_layers": 1,
                    "hidden_size": 32,
                    "bidirectional": True,
                },
                "dropout": 0.1,
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


def test_text_classification(tmp_path, pipeline_dict, train_valid_dataset):
    """Apart from a well specified training, this also tests the vocab creation!"""
    seed_everything(43)

    pl = Pipeline.from_config(pipeline_dict)
    train_ds = train_valid_dataset[0]
    valid_ds = train_valid_dataset[1]

    vocab_config = VocabularyConfiguration(max_vocab_size={"word": 50})
    trainer_config = TrainerConfiguration(
        batch_size=64,
        optimizer={"type": "adam", "lr": 0.01},
        max_epochs=5,
        default_root_dir=str(tmp_path),
        gpus=0,  # turn off gpus even if available
    )

    trainer = Trainer(
        pipeline=pl,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        trainer_config=trainer_config,
        vocab_config=vocab_config,
    )

    trainer.fit(tmp_path / "output")

    assert pl.vocab.get_vocab_size(WordFeatures.namespace) == 52
    assert pl.vocab.get_vocab_size(CharFeatures.namespace) == 83

    assert pl.num_trainable_parameters == 22070

    evaluation = trainer.test(valid_ds, batch_size=16)

    # Reminder: the value depends on the batch_size!
    assert evaluation["test_loss"] == pytest.approx(0.7404146790504456, abs=0.003)

    Pipeline.from_pretrained(str(tmp_path / "output" / "model.tar.gz"))

    assert pl.vocab.get_vocab_size(WordFeatures.namespace) == 52
    assert pl.vocab.get_vocab_size(CharFeatures.namespace) == 83
