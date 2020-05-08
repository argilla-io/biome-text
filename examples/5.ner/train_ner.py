from biome.text import Pipeline, VocabularyConfiguration
from biome.text import TrainerConfiguration
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":
    pl = Pipeline.from_yaml("configs/char_gru_token_classifier.yml")
    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))
    pl.train(
        output="experiment",
        trainer=trainer,
        training="configs/train.data.yml",
        validation="configs/validation.data.yml",
        extend_vocab=VocabularyConfiguration(sources=["configs/train.data.yml"]),
    )
