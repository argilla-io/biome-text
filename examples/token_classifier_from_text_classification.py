from biome.text import Pipeline
from biome.text.modules.heads import TokenClassification

if __name__ == "__main__":
    pipe = Pipeline.from_pretrained("1.text_classifier/experiment/model.tar.gz")
    print(pipe.predict(text="This is a text blo blo blo blo!!"))

    pipe.set_head(TokenClassification, labels=["B", "O", "I-PER", "I-ORG"])
    print(pipe.predict(text="This is a text blo blo blo blo!!".split(" ")))
