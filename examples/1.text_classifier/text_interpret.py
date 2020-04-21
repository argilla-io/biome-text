from biome.text.api_new import Pipeline

if __name__ == "__main__":
    trained_pl = Pipeline.from_binary("experiment/model.tar.gz")
    print(trained_pl.explain(text="The simple file"))
