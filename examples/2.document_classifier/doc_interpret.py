from biome.text import Pipeline

if __name__ == "__main__":
    trained_pl = Pipeline.from_pretrained("experiment/model.tar.gz")

    document = [
        "this is my email subject",
        "A very long body. I have many sentences. I am writing very boring, long emails.",
    ]
    print(trained_pl.explain(document=document))
    trained_pl.serve()
