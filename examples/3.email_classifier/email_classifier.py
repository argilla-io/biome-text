from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration
from biome.text.data import DataSource
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":

    pl = Pipeline.from_yaml("email_classifier.yaml")
    pl.head.extend_labels(["a", "b"])
    print(
        pl.predict(
            subject="Header main. This is a test body!!!",
            body="The next phrase is here",
        )
    )
    training_ds = DataSource.from_yaml("train.data.yml")
    validation_ds = DataSource.from_yaml("validation.data.yml")
    pl.create_vocabulary(
        VocabularyConfiguration(sources=[training_ds, validation_ds])
    )

    trainer = TrainerConfiguration(**yaml_to_dict("trainer.yml"))

    pl.train(
        output="experiment",
        trainer=trainer,
        training=training_ds,
        validation=validation_ds,
    )

    trained = Pipeline.from_pretrained("experiment/model.tar.gz")
    trained.predict(
        subject="Header main. This is a test body!!!", body="The next phrase is here"
    )
    trained.head.extend_labels(["other"])
    trained.explore(data_source=validation_ds)
