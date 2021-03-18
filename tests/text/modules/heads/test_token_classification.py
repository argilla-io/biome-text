from typing import Dict

import pandas as pd
import pytest
from allennlp.data import Batch

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import vocabulary
from biome.text.modules.heads.task_prediction import Entity
from biome.text.modules.heads.task_prediction import TokenClassificationPrediction


@pytest.fixture
def training_dataset() -> Dataset:
    df = pd.DataFrame(
        {
            "text": [
                "This is a simple NER test",
                "This is a simple NER test with misaligned spans",
                "No NER here",
            ],
            "entities": [
                [{"start": 17, "end": 20, "label": "NER"}],
                [{"start": 17, "end": 22, "label": "NER"}],
                [],
            ],
        }
    )

    return Dataset.from_pandas(df)


@pytest.fixture
def pipeline_dict() -> Dict:
    pipeline_dict = {
        "name": "biome-bimpm",
        "features": {"word": {"embedding_dim": 2}},
        "head": {
            "type": "TokenClassification",
            "labels": ["NER"],
            "label_encoding": "BIOUL",
        },
    }

    return pipeline_dict


@pytest.fixture
def trainer_dict() -> Dict:
    trainer_dict = {
        "num_epochs": 1,
        "batch_size": 1,
        "optimizer": {"type": "adam", "lr": 0.01},
        "cuda_device": -1,
    }

    return trainer_dict


def test_tokenization_with_blank_tokens(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)
    predictions = pipeline.predict(text="Test this text \n \n", entities=[])
    assert len(predictions["tags"][0]) == 4


def test_train(pipeline_dict, training_dataset, trainer_dict, tmp_path):
    pipeline = Pipeline.from_config(pipeline_dict)

    assert pipeline.output == ["entities", "tags"]

    assert pipeline.head.span_labels == ["NER"]
    assert pipeline.head.labels == ["B-NER", "I-NER", "U-NER", "L-NER", "O"]

    pipeline.train(
        output=str(tmp_path / "ner_experiment"),
        trainer=TrainerConfiguration(**trainer_dict),
        training=training_dataset,
    )


class TestMakeTaskPrediction:
    def test_pretokenized_input(self, pipeline_dict):
        pipeline = Pipeline.from_config(pipeline_dict)
        output = self._input_top_k2(pipeline)
        expected_output = TokenClassificationPrediction(
            tags=[["O", "O", "O", "U-NER"], ["O", "B-NER", "I-NER", "L-NER"]],
            entities=[
                [Entity(start_token=3, end_token=4, label="NER")],
                [Entity(start_token=1, end_token=4, label="NER")],
            ],
            scores=[2, 1],
        )

        assert output == expected_output

    def test_untokenized_input(self, pipeline_dict):
        pipeline = Pipeline.from_config(pipeline_dict)
        output = self._input_top_k2(pipeline, pretokenized=False)
        expected_output = TokenClassificationPrediction(
            tags=[["O", "O", "O", "U-NER"], ["O", "B-NER", "I-NER", "L-NER"]],
            entities=[
                [Entity(start_token=3, end_token=4, label="NER", start=10, end=14)],
                [Entity(start_token=1, end_token=4, label="NER", start=5, end=14)],
            ],
            scores=[2, 1],
        )

        assert output == expected_output

    @staticmethod
    def _input_top_k2(pipeline, pretokenized=True):
        raw_text = ["this", "is", "a", "test"] if pretokenized else "this is a test"
        tag_idx_sequence = [
            vocabulary.index_for_label(pipeline.backbone.vocab, tag)
            for tag in ["O", "O", "O", "U-NER"]
        ]
        tag_idx_sequence2 = [
            vocabulary.index_for_label(pipeline.backbone.vocab, tag)
            for tag in ["O", "B-NER", "I-NER", "L-NER"]
        ]
        viterbi_paths = [(tag_idx_sequence, 2), (tag_idx_sequence2, 1)]
        single_forward_output = dict(
            viterbi_paths=viterbi_paths,
            raw_text=raw_text,
        )

        return pipeline.head._make_task_prediction(single_forward_output, None)


def test_preserve_pretokenization(
    pipeline_dict, training_dataset, trainer_dict, tmp_path
):
    pipeline = Pipeline.from_config(pipeline_dict)
    tokens = ["test", "this", "pre tokenized", "text"]
    prediction = pipeline.predict(tokens)
    assert len(prediction["tags"][0]) == len(tokens)


def test_metrics(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)
    instance = pipeline.head.featurize(text="test this".split(), tags=["U-NER", "O"])
    batch = Batch([instance])
    batch.index_instances(pipeline.vocab)

    pipeline.head.forward(**batch.as_tensor_dict())
    # validation metric should have never been called
    assert pipeline.head._metrics.get_dict()["accuracy"].total_count == 2
    assert pipeline.head._metrics.get_dict(is_train=False)["accuracy"].total_count == 0

    train_metrics = pipeline.head.get_metrics(reset=True)
    expected_metric_names = ["accuracy"] + [
        f"{metric}-{label}"
        for metric in ["precision", "recall", "f1-measure"]
        for label in ["NER", "overall"]
    ]
    print(train_metrics)
    assert all(name in train_metrics for name in expected_metric_names)

    pipeline.head.training = False
    pipeline.head.forward(**batch.as_tensor_dict())
    # training metric should have never been called after its reset
    assert pipeline.head._metrics.get_dict()["accuracy"].total_count == 0
    assert pipeline.head._metrics.get_dict(is_train=False)["accuracy"].total_count == 2

    valid_metrics = pipeline.head.get_metrics()
    assert all(name in valid_metrics for name in expected_metric_names)
