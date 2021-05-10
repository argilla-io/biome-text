import pytest
from allennlp.data import Batch
from numpy.testing import assert_allclose

from biome.text import Pipeline
from biome.text.modules.heads.task_prediction import Attribution
from biome.text.modules.heads.task_prediction import TextClassificationPrediction


@pytest.fixture
def pipeline() -> Pipeline:
    labels = ["a", "b", "c", "d", "e", "f"]
    return Pipeline.from_config(
        {
            "name": "test_text_classification",
            "head": {"type": "TextClassification", "labels": labels},
        }
    )


def test_make_task_prediction(pipeline):
    instance = pipeline.head.featurize("test this sentence")
    forward_output = pipeline._model.forward_on_instances([instance])

    prediction = pipeline.head._make_task_prediction(forward_output[0], None)

    assert isinstance(prediction, TextClassificationPrediction)
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

    assert all([isinstance(attribution, Attribution) for attribution in attributions])
    assert len(attributions) == 3
    assert all([attr.field == "text" for attr in attributions])
    assert all([isinstance(attr.attribution, float) for attr in attributions])
    assert attributions[1].start == 5 and attributions[1].end == 9


def test_metrics(pipeline):
    instance = pipeline.head.featurize(text="test this", label="a")
    batch = Batch([instance])
    batch.index_instances(pipeline.vocab)

    pipeline.head.forward(**batch.as_tensor_dict())
    # validation metric should have never been called
    assert pipeline.head._metrics.get_dict()["accuracy"].total_count == 1
    assert pipeline.head._metrics.get_dict(is_train=False)["accuracy"].total_count == 0

    train_metrics = pipeline.head.get_metrics(reset=True)
    expected_metric_names = (
        ["accuracy"]
        + [
            f"{label}/{metric}"
            for label in ["micro", "macro"]
            for metric in ["precision", "recall", "fscore"]
        ]
        + [
            f"_{metric}/{label}"
            for metric in ["precision", "recall", "fscore"]
            for label in ["a", "b", "c", "d", "e", "f"]
        ]
    )
    assert all(name in train_metrics for name in expected_metric_names)

    pipeline.head.training = False
    pipeline.head.forward(**batch.as_tensor_dict())
    # training metric should have never been called after its reset
    assert pipeline.head._metrics.get_dict()["accuracy"].total_count == 0
    assert pipeline.head._metrics.get_dict(is_train=False)["accuracy"].total_count == 1

    valid_metrics = pipeline.head.get_metrics()
    assert all(name in valid_metrics for name in expected_metric_names)
