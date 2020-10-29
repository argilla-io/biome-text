import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from biome.text import Pipeline, Dataset, TrainerConfiguration, VocabularyConfiguration
from biome.text.configuration import CharFeatures
from biome.text.configuration import WordFeatures


@pytest.fixture
def train_valid_data_source() -> Tuple[Dataset, Dataset]:
    """Returns both training and validation datasets"""

    resources_path = Path(__file__).parent.parent / "resources" / "data"
    training_ds = Dataset.from_csv(paths=str(resources_path / "business.cat.2k.train.csv"))
    validation_ds = Dataset.from_csv(paths=str(resources_path / "business.cat.2k.valid.csv"))

    return training_ds, validation_ds


@pytest.fixture
def pipeline_dict() -> dict:
    """Pipeline config dict. You need to update the labels!"""

    pipeline_dictionary = {
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

    return pipeline_dictionary


@pytest.fixture
def trainer_dict() -> dict:
    """Returns the trainer dictionary"""

    return {
        "batch_size": 64,
        "num_epochs": 5,
        "optimizer": {"type": "adam", "lr": 0.01},
        "cuda_device": -1,
    }


def test_text_classification(tmp_path, pipeline_dict, trainer_dict, train_valid_data_source):
    """Apart from a well specified training, this also tests the vocab creation!"""

    random.seed(42)
    np.random.seed(422)
    torch.manual_seed(4222)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4222)

    pl = Pipeline.from_config(pipeline_dict)
    train_ds = train_valid_data_source[0].to_instances(pipeline=pl)
    valid_ds = train_valid_data_source[1].to_instances(pipeline=pl)
    trainer = TrainerConfiguration(**trainer_dict)
    vocab = VocabularyConfiguration(sources=[train_ds], max_vocab_size={"word": 50})

    pl.create_vocabulary(vocab)

    assert pl._model.vocab.get_vocab_size(WordFeatures.namespace) == 52
    assert pl._model.vocab.get_vocab_size(CharFeatures.namespace) == 83

    output = tmp_path / "output"

    pl.train(
        output=str(output), trainer=trainer, training=train_ds, validation=valid_ds
    )

    assert pl.num_trainable_parameters == 22070

    with (output / "metrics.json").open() as file:
        metrics = json.load(file)

    # It may fail in some systems
    assert metrics["training_loss"] == pytest.approx(0.670, abs=0.003)

    # Test vocab from a pretrained file
    pl = Pipeline.from_pretrained(str(output / "model.tar.gz"))

    assert pl._model.vocab.get_vocab_size(WordFeatures.namespace) == 52
    assert pl._model.vocab.get_vocab_size(CharFeatures.namespace) == 83