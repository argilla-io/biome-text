from biome.text import Pipeline, VocabularyConfiguration
from biome.text import TrainerConfiguration
from biome.text.data import DataSource
from biome.text.helpers import yaml_to_dict


if __name__ == "__main__":
    pl = Pipeline.from_yaml("configs/text_classifier.yml")

    training_ds = DataSource.from_yaml("configs/train.data.yml")
    validation_ds = DataSource.from_yaml("configs/val.data.yml")
    pl.create_vocabulary(VocabularyConfiguration(sources=[training_ds], min_count={"words": 12}))

    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))

    pl.train(
        output="experiment_text_classifier",
        trainer=trainer,
        training=training_ds,
        validation=validation_ds,
    )
