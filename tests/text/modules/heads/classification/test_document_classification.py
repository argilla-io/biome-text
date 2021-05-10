import pytest
from numpy.testing import assert_allclose

from biome.text import Pipeline
from biome.text.modules.heads.task_prediction import Attribution
from biome.text.modules.heads.task_prediction import DocumentClassificationPrediction


@pytest.fixture
def pipeline() -> Pipeline:
    labels = ["a", "b", "c", "d", "e", "f"]
    return Pipeline.from_config(
        {
            "name": "test_document_classification",
            "tokenizer": {"segment_sentences": False},
            "head": {"type": "DocumentClassification", "labels": labels},
        }
    )


@pytest.mark.parametrize(
    "segment_sentences, input, output",
    [
        (False, "one sentence. two sentence", (1, 5)),
        (True, "one sentence. two sentence", (2, 3)),
        (False, ["one sentence. two sentence", "test"], (2, 5)),
        (True, ["one sentence. two sentence", "test"], (3, 3)),
        (False, {"one": "one sentence. two sentence", "two": "test"}, (2, 5)),
        (True, {"one": "one sentence. two sentence", "two": "test"}, (3, 3)),
    ],
)
def test_tokenization_of_different_input(pipeline, segment_sentences, input, output):
    pipeline = Pipeline.from_config(
        {
            "name": "test_document_classification",
            "tokenizer": {"segment_sentences": segment_sentences},
            "head": {"type": "DocumentClassification", "labels": "a"},
        }
    )
    instance = pipeline.head.featurize(input)
    tokens = pipeline.head._extract_tokens(instance)

    assert len(tokens) == output[0]
    assert len(tokens[0]) == output[1]


def test_make_task_prediction(pipeline):
    instance = pipeline.head.featurize("test this sentence")
    forward_output = pipeline._model.forward_on_instances([instance])

    prediction = pipeline.head._make_task_prediction(forward_output[0], None)

    assert isinstance(prediction, DocumentClassificationPrediction)
    assert isinstance(prediction.labels, list) and isinstance(
        prediction.probabilities, list
    )
    assert len(prediction.labels) == len(prediction.probabilities) == 6
    # check descending order
    assert_allclose(
        sorted(prediction.probabilities, reverse=True), prediction.probabilities
    )
    assert all([isinstance(label, str) for label in prediction.labels])
    assert set(pipeline.head.labels) == set(prediction.labels)
    assert all([isinstance(prob, float) for prob in prediction.probabilities])


def test_compute_attributions(pipeline):
    instance = pipeline.head.featurize("test this sentence")
    pipeline.model.eval()
    forward_output = pipeline.model.forward_on_instances([instance])

    attributions = pipeline.head._compute_attributions(
        forward_output[0], instance, n_steps=1
    )

    assert isinstance(attributions, list) and isinstance(attributions[0], list)
    assert len(attributions) == 1 and len(attributions[0]) == 3
    assert all(
        [isinstance(attribution, Attribution) for attribution in attributions[0]]
    )
    assert all([attr.field == "text" for attr in attributions[0]])
    assert all([isinstance(attr.attribution, float) for attr in attributions[0]])
    assert attributions[0][1].start == 5 and attributions[0][1].end == 9
