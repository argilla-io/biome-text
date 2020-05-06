from biome.text.api_new import Pipeline
from biome.text.api_new.configuration import TrainerConfiguration
from biome.text.api_new.helpers import yaml_to_dict

if __name__ == "__main__":
    pl = Pipeline.from_file("configs/text_classifier.yml")
    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))
    trained_pl = pl.train(
        output="experiment_text_classifier",
        trainer=trainer,
        training="configs/train.data.yml",
        validation="configs/val.data.yml",
    )
