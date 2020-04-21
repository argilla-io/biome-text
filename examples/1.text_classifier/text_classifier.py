from biome.text.api_new import Pipeline, VocabularyConfiguration

if __name__ == "__main__":
    train = "train.data.yml"
    validation = "validation.data.yml"

    pl = Pipeline.from_file(
        "text_classifier.yaml",
        vocab_config=VocabularyConfiguration(sources=[train, validation]),
    )

    print(pl.predict(text="Header main. This is a test body!!!"))

    trained_pl = pl.train(
        output="experiment",
        trainer="trainer.yml",
        training=train,
        validation=validation,
    )

    trained_pl.predict(text="Header main; This is a test body!!!")
    trained_pl.head.extend_labels(["other"])
    trained_pl.explore(
        explore_id="test-trained", ds_path="validation.data.yml", explain=True,
    )

    trained_pl = trained_pl.train(
        output="experiment.v2",
        trainer="trainer.yml",
        training=train,
        validation=validation,
    )

    trained_pl.predict(text="Header main. This is a test body!!!")

    pl.head.extend_labels(["yes", "no"])
    pl.explore(ds_path="validation.data.yml")
