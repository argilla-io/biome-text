from biome.text.api_new import Pipeline

if __name__ == "__main__":
    trained_pl = Pipeline.from_binary("experiment/model.tar.gz")

    document = [
        "Prrrt",
        "The simple file enormeous sentence length",
        "Another sentence",
    ]
    another_document = ["Uno", "Uno dos", "uno dos tres"]
    print(trained_pl.explain(document=another_document))
