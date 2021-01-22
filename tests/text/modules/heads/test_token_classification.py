from typing import Dict

import pandas as pd
import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import vocabulary
from biome.text.modules.heads import TaskPrediction
from biome.text.modules.heads.task_prediction import Entity
from biome.text.modules.heads.task_prediction import Token
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


def test_default_explain(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)

    prediction = pipeline.explain("This is a simple text")
    assert prediction["explain"]
    assert len(prediction["explain"]["text"]) == len(prediction["tags"][0][0])
    # enable training mode for generate instances with tags
    pipeline.head.train()

    prediction = pipeline.explain(text="This is a simple text", entities=[])
    assert len(prediction["explain"]["tags"]) == len(prediction["explain"]["text"])

    for label in prediction["explain"]["tags"]:
        assert "label" in label
        assert "token" in label


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


class TestMakeTaskOutput:
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
            tokens=[
                Token(end=4, start=0, text="this"),
                Token(end=7, start=5, text="is"),
                Token(end=9, start=8, text="a"),
                Token(end=14, start=10, text="test"),
            ],
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

        return pipeline.head.make_task_prediction(single_forward_output)


def test_preserve_pretokenization(
    pipeline_dict, training_dataset, trainer_dict, tmp_path
):
    pipeline = Pipeline.from_config(pipeline_dict)
    tokens = ["test", "this", "pre tokenized", "text"]
    prediction = pipeline.predict(tokens)
    assert len(prediction["tags"][0]) == len(tokens)
