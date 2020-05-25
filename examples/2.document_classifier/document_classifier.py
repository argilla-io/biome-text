from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration
from biome.text.data import DataSource
from biome.text.helpers import yaml_to_dict

if __name__ == "__main__":
    pl = Pipeline.from_yaml("document_classifier.yaml", vocab_path="not_found_folder")
    print(f"Pipeline parameters: {pl.trainable_parameter_names}")
    print(f"Trainable parameters: {pl.trainable_parameters}")
    print(
        pl.predict(
            document=["Header main. This is a test body!!!", "The next phrase is here"]
        )
    )
    # See how is running record tokenization
    pl.explain(
        document=dict(
            record1="this is the field",
            record2="The next segment. This is another sentence again",
        )
    )

    trainer = TrainerConfiguration(**yaml_to_dict("trainer.yml"))
    pl.create_vocabulary(
        VocabularyConfiguration(
            sources=[DataSource.from_yaml("train.data.yml")], min_count={"words": 10}
        )
    )
    pl.train(
        output="experiment",
        trainer=trainer,
        training="train.data.yml",
        validation="validation.data.yml",
    )

    pl = Pipeline.from_pretrained("experiment/model.tar.gz")
    pl.predict(
        document=["Header main. This is a test body!!!", "The next phrase is here"]
    )
    pl.explore(data_source="validation.data.yml")

    trained_pl = Pipeline.from_pretrained("experiment/model.tar.gz")
    trained_pl.explore(data_source="validation.data.yml")
