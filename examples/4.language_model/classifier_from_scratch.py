from biome.text.api_new import Pipeline

if __name__ == "__main__":
    pl = Pipeline.from_file("configs/text_classifier.yml")

    trained_pl = pl.train(
        output="experiment_text_classifier",
        trainer="configs/trainer.yml",
        training="configs/train.data.yml",
        validation="configs/val.data.yml",
    )
