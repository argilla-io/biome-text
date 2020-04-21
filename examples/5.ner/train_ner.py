from biome.text.api_new import Pipeline

if __name__ == "__main__":
    pl = Pipeline.from_file("configs/char_gru_token_classifier.yml")

    trained_pl = pl.train(
        output="experiment",
        trainer="configs/trainer.yml",
        training="configs/train.data.yml",
        validation="configs/validation.data.yml",
    )
