from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text.helpers import yaml_to_dict
from biome.text.modules.heads import TextClassification
from biome.text.data import DataSource

if __name__ == "__main__":
    # load an existing pre-trained model
    pipe = Pipeline.from_pretrained("configs/experiment_lm/model.tar.gz")

    labels = [
        "Allgemeinärzte",
        "Apotheken",
        "Architekturbüros",
        "Autowerkstätten",
        "Dienstleistungen",
        "Edv",
        "Elektriker",
        "Elektrotechnik",
        "Friseure",
        "Gebrauchtwagen",
        "Handelsvermittler Und -vertreter",
        "Hotels",
        "Maler",
        "Physiotherapie",
        "Restaurants",
        "Sanitärinstallationen",
        "Tiefbau",
        "Unternehmensberatungen",
        "Vereine",
        "Vermittlungen",
        "Versicherungsvermittler",
        "Werbeagenturen",
    ]

    pipe.set_head(TextClassification, pooler={"type": "boe"}, labels=labels)
    trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))
    pipe.train(
        output="text_classifier_fine_tuned",
        trainer=trainer,
        training=DataSource.from_yaml("configs/train.data.yml"),
        validation=DataSource.from_yaml("configs/val.data.yml"),
    )
