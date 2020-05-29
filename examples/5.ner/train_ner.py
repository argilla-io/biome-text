from biome.text import Pipeline, VocabularyConfiguration
from biome.text import TrainerConfiguration
from biome.text.data import DataSource
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":
    training_ds = DataSource.from_yaml("configs/train.data.yml")
    validation_ds = DataSource.from_yaml("configs/val.data.yml")

    pl = Pipeline.from_yaml("configs/char_gru_token_classifier.yml")
    pl.create_vocabulary(VocabularyConfiguration(sources=[training_ds]))
    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))

    pl.train(
        output="experiment",
        trainer=trainer,
        training=training_ds,
        validation=validation_ds,
    )
