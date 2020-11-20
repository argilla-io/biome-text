from biome.text import Pipeline


def test_classification_output():
    pipeline = Pipeline.from_config(
        {
            "name": "test_text_classification",
            "head": {"type": "TextClassification", "labels": ["a", "b"]},
        }
    )
    prediction = pipeline.predict(text="check")

    assert len(prediction) == 1 and "labels" in prediction
    assert {"a", "b"} == {label for label, prob in prediction["labels"]}
    assert isinstance(prediction["labels"], list) and len(prediction["labels"]) == 2
    assert (
        isinstance(prediction["labels"][0], tuple) and len(prediction["labels"][0]) == 2
    )
    assert isinstance(prediction["labels"][0][0], str) and isinstance(
        prediction["labels"][0][1], float
    )
