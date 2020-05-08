from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":

    pl = Pipeline.from_file("email_classifier.yaml")
    pl.head.extend_labels(["a", "b"])
    print(
        pl.predict(
            subject="Header main. This is a test body!!!",
            body="The next phrase is here",
        )
    )

    trainer = TrainerConfiguration(**yaml_to_dict("trainer.yml"))
    trained_pl = pl.train(
        output="experiment",
        trainer=trainer,
        training="train.data.yml",
        validation="validation.data.yml",
        extend_vocab=VocabularyConfiguration(sources=["validation.data.yml"]),
    )

    trained_pl.predict(
        subject="Header main. This is a test body!!!", body="The next phrase is here"
    )
    trained_pl.head.extend_labels(["other"])
    trained_pl.explore(ds_path="validation.data.yml")
