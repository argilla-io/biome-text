from biome.text.api_new import Pipeline, VocabularyConfiguration

if __name__ == "__main__":
    train = "configs/train.data.yml"
    validation = "configs/val.data.yml"

    pl = Pipeline.from_file(
        "configs/language_model.yml",
        vocab_config=VocabularyConfiguration(sources=[train, validation]),
    )

    trained_pl = pl.train(
        output="configs/experiment_lm",
        trainer="configs/trainer.yml",
        training=train,
        validation=validation,
    )
