from biome.text import Pipeline

if __name__ == "__main__":

    pipeline = Pipeline.from_pretrained("experiment/model.tar.gz")
    print(
        pipeline.explain(
            subject="Header main. This is a test body!!!",
            body="The next phrase is here",
        )
    )
    pipeline.serve()
