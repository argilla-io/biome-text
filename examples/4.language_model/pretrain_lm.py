from biome.text import Pipeline, VocabularyConfiguration, TrainerConfiguration
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":
    train = "configs/train.data.yml"
    validation = "configs/val.data.yml"

    pl = Pipeline.from_yaml("configs/language_model.yml")
    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))
    pl.train(
        output="configs/experiment_lm",
        trainer=trainer,
        training=train,
        validation=validation,
        extend_vocab=VocabularyConfiguration(
            sources=[train, validation], min_count={"words": 12}
        ),
    )
