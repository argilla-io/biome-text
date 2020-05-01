from biome.text.api_new import Pipeline, VocabularyConfiguration
from biome.text.api_new.configuration import TrainerConfiguration
from biome.text.api_new.helpers import yaml_to_dict

if __name__ == "__main__":
    train = "configs/train.data.yml"
    validation = "configs/val.data.yml"

    pl = Pipeline.from_file(
        "configs/language_model.yml",
        vocab_config=VocabularyConfiguration(sources=[train, validation]),
    )
    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))
    trained_pl = pl.train(
        output="configs/experiment_lm",
        trainer=trainer,
        training=train,
        validation=validation,
    )
