import os

from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration

from biome.text.data import DataSource
from biome.text.featurizer import WordFeatures
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":
    train = "toxic_train.yml"
    validation = "toxic_validation.yml"
    training_folder = "experiment"

    pl = Pipeline.from_yaml("text_classifier-multi.yaml")

    pl.create_vocabulary(
        VocabularyConfiguration(
            sources=[DataSource.from_yaml(path) for path in [train, validation]],
            min_count={WordFeatures.namespace: 5},
        )
    )

    trainer_configuration = TrainerConfiguration(**yaml_to_dict("trainer.yml"))
    trainer_configuration.data_bucketing = False

    pl.train(
        output=training_folder,
        trainer=trainer_configuration,
        training=train,
        validation=train,
    )
