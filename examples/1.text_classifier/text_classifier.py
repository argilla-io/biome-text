import os

from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration

from biome.text.data import DataSource
from biome.text.featurizer import WordFeatures
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":
    train = DataSource.from_yaml("train.data.yml")
    validation = DataSource.from_yaml("validation.data.yml")
    training_folder = "experiment"

    pl = Pipeline.from_yaml("text_classifier.yaml")

    print(pl.predict(text="Header main. This is a test body!!!"))

    pl.create_vocabulary(
        VocabularyConfiguration(
            sources=[train, validation],
            min_count={WordFeatures.namespace: 5},
        )
    )

    trainer_configuration = TrainerConfiguration(**yaml_to_dict("trainer.yml"))
    trainer_configuration.data_bucketing = False

    pl.train(
        output=training_folder,
        trainer=trainer_configuration,
        training=train,
        validation=validation,
    )

    trained_pl = Pipeline.from_pretrained(os.path.join(training_folder, "model.tar.gz"))
    trained_pl.predict(text="Header main; This is a test body!!!")
    trained_pl.head.extend_labels(["other"])
    trained_pl.explore(
        explore_id="test-trained", data_source=validation, explain=True
    )

    trainer_configuration.batch_size = 8
    trainer_configuration.data_bucketing = True
    trained_pl.train(
        output="experiment.v2",
        trainer=trainer_configuration,
        training=train,
        validation=validation,
    )

    trained_pl = Pipeline.from_pretrained(os.path.join("experiment.v2", "model.tar.gz"))
    trained_pl.predict(text="Header main. This is a test body!!!")

    pl.head.extend_labels(["yes", "no"])
    pl.explore(data_source=validation)
