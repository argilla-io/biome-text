from allennlp.data import Vocabulary

from biome.text.metrics import Metrics


def test_metrics():
    metrics = Metrics(
        accuracy={"type": "categorical_accuracy"},
        f1={
            "type": "span_f1",
            "vocabulary": Vocabulary.empty(),
        },
    )

    # Check that training and validation metrics are different instances
    assert (
        metrics.get_dict()["accuracy"]
        is not metrics.get_dict(is_train=False)["accuracy"]
    )
    # Check if we share the same vocab
    assert (
        metrics.get_dict()["f1"]._label_vocabulary
        is metrics.get_dict(is_train=False)["f1"]._label_vocabulary
    )
