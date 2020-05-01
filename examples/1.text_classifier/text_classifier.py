from biome.text.api_new import Pipeline, VocabularyConfiguration
from biome.text.api_new.configuration import TrainerConfiguration
from biome.text.api_new.helpers import yaml_to_dict

if __name__ == "__main__":
    train = "train.data.yml"
    validation = "validation.data.yml"

    pl = Pipeline.from_file(
        "text_classifier.yaml",
        vocab_config=VocabularyConfiguration(sources=[train, validation]),
    )

    print(pl.predict(text="Header main. This is a test body!!!"))

    trainer_configuration = TrainerConfiguration(**yaml_to_dict("trainer.yml"))
    trainer_configuration.data_bucketing = False
    trained_pl = pl.train(
        output="experiment",
        trainer=trainer_configuration,
        training=train,
        validation=validation,
    )

    trained_pl.predict(text="Header main; This is a test body!!!")
    trained_pl.head.extend_labels(["other"])
    trained_pl.explore(
        explore_id="test-trained", ds_path="validation.data.yml", explain=True
    )

    trainer_configuration.batch_size = 8
    trainer_configuration.data_bucketing = True
    trained_pl = trained_pl.train(
        output="experiment.v2",
        trainer=trainer_configuration,
        training=train,
        validation=validation,
    )

    trained_pl.predict(text="Header main. This is a test body!!!")

    pl.head.extend_labels(["yes", "no"])
    pl.explore(ds_path="validation.data.yml")
