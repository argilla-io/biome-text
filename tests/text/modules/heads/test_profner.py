import itertools

import pytest
from allennlp.data import Batch
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data.fields import TextField

from biome.text import Dataset
from biome.text import Pipeline


@pytest.fixture
def dataset() -> Dataset:
    data = {
        "tokens": [
            ["Test", "this"],
            ["superlongword", "for", "the", "transformer", "tokenizer"],
        ],
        "tags": [["O", "O"], ["B-TEST", "O", "B-TEST", "I-TEST", "I-TEST"]],
        "labels": ["0", "1"],
    }

    return Dataset.from_dict(data)


@pytest.fixture
def profnert(dataset) -> dict:
    model_name = "sshleifer/tiny-distilbert-base-cased"
    model_name = "prajjwal1/bert-tiny"

    return {
        "name": "test_profnert",
        "features": {
            "transformers": {
                "model_name": model_name,
                "trainable": True,
            }
        },
        "head": {
            "type": "ProfNerT",
            "classification_labels": dataset.unique("labels"),
            "classification_pooler": {
                "type": "bert_pooler",
                "pretrained_model": model_name,
                "requires_grad": True,
                "dropout": 0.1,
            },
            "ner_tags": list(set(itertools.chain.from_iterable(dataset["tags"]))),
            "ner_tags_encoding": "BIO",
            "transformers_model": model_name,
            "dropout": 0.0,
            "ner_feedforward": {
                "activations": ["relu"],
                "dropout": [0],
                "hidden_dims": [32],
                "num_layers": 1,
            },
        },
    }


@pytest.fixture
def profner(dataset) -> dict:
    return {
        "name": "test_profner",
        "features": {
            "word": {
                "embedding_dim": 300,
            },
        },
        "encoder": {
            "type": "lstm",
            "num_layers": 1,
            "bidirectional": True,
            "hidden_size": 128,
        },
        "head": {
            "type": "ProfNer",
            "classification_labels": dataset.unique("labels"),
            "classification_pooler": {
                "type": "gru",
                "num_layers": 1,
                "bidirectional": True,
                "hidden_size": 64,
            },
            "ner_tags": list(set(itertools.chain.from_iterable(dataset["tags"]))),
            "ner_tags_encoding": "BIO",
            "dropout": 0.0,
            "ner_feedforward": {
                "activations": ["relu"],
                "dropout": [0],
                "hidden_dims": [32],
                "num_layers": 1,
            },
        },
    }


@pytest.fixture(params=["profner", "profnert"])
def pipeline_dict(request, profner, profnert):
    return {"profner": profner, "profnert": profnert}[request.param]


def test_profner(pipeline_dict: dict, dataset: Dataset, tmp_path):
    print(pipeline_dict)
    pipeline = Pipeline.from_config(pipeline_dict)
    pipeline.train(output=str(tmp_path / "test_output"), training=dataset)
    predictions = pipeline.predict(
        batch=[
            {"tokens": ["Test", "this"]},
            {"tokens": ["superlongword", "for", "the", "transformer", "tokenizer"]},
        ],
        add_tokens=True,
    )
    print(predictions)


def test_null():
    tf = TextField("", {})
    instance = Instance({"tf": tf})
    vocab = Vocabulary.from_instances([instance])
    batch = Batch([instance])
    batch.index_instances(vocab)

    print(batch.as_tensor_dict())
