from biome.text.api_new import Pipeline
from biome.text.api_new.configuration import TrainerConfiguration, VocabularyConfiguration
from biome.text.api_new.helpers import yaml_to_dict

if __name__ == "__main__":

    pl = Pipeline.from_file("document_classifier.yaml")
    print(f"Pipeline parameters: {pl.trainable_parameter_names}")
    print(f"Trainable parameters: {pl.trainable_parameters}")
    print(
        pl.predict(
            document=["Header main. This is a test body!!!", "The next phrase is here"]
        )
    )

    trainer = TrainerConfiguration(**yaml_to_dict("trainer.yml"))
    trained_pl = pl.train(
        output="experiment",
        trainer=trainer,
        training="train.data.yml",
        validation="validation.data.yml",
        verbose=True,
    )

    trained_pl.predict(
        document=["Header main. This is a test body!!!", "The next phrase is here"]
    )
    trained_pl.explore(ds_path="validation.data.yml")

    trained_pl = trained_pl.train(
        output="experiment.v2",
        trainer=trainer,
        training="train.data.yml",
        validation="validation.data.yml",
    )

    trained_pl.predict(
        document=["Header main. This is a test body!!!", "The next phrase is here"]
    )

    pl.head.extend_labels(["yes", "no"])
    pl.explore(
        explore_id="test-document-explore", ds_path="validation.data.yml",
    )

    # pl.serve()
