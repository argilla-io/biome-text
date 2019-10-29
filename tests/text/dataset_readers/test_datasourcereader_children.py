from inspect import signature

import pytest
from allennlp.common.util import import_submodules
from allennlp.data import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model, BertForClassification

from biome.text.dataset_readers import (
    SequenceClassifierDatasetReader,
    SequencePairClassifierDatasetReader,
)
from biome.text.models import (
    SequenceClassifier,
    SequencePairClassifier,
    SimilarityClassifier,
)

import_submodules("biome.text")


@pytest.mark.parametrize(
    "name, model, dataset_reader",
    [
        ("sequence_classifier", SequenceClassifier, SequenceClassifierDatasetReader),
        (
            "bert_for_classification",
            BertForClassification,
            SequenceClassifierDatasetReader,
        ),
        (
            "sequence_pair_classifier",
            SequencePairClassifier,
            SequencePairClassifierDatasetReader,
        ),
        (
            "similarity_classifier",
            SimilarityClassifier,
            SequencePairClassifierDatasetReader,
        ),
    ],
)
def test_name_consistency(name, model, dataset_reader):
    """One of our design choices: each model has its own dataset reader with the same name"""
    assert model is Model.by_name(name)
    assert dataset_reader is DatasetReader.by_name(name)


@pytest.mark.parametrize(
    "dataset_reader, model, text_input",
    [
        (
            SequenceClassifierDatasetReader,
            SequenceClassifier,
            {"tokens": "a", "label": "b"},
        ),
        (
            SequenceClassifierDatasetReader,
            BertForClassification,
            {"tokens": "a", "label": "b"},
        ),
        (
            SequencePairClassifierDatasetReader,
            SequencePairClassifier,
            {"record1": "a", "record2": "b", "label": "c"},
        ),
        (
            SequencePairClassifierDatasetReader,
            SimilarityClassifier,
            {"record1": "a", "record2": "b", "label": "c"},
        ),
    ],
)
def test_signature_consistency(dataset_reader, model, text_input):
    """The output of the `text_to_instance` method has to match the forward signature of the model's forward method!"""
    reader = dataset_reader(
        tokenizer=WordTokenizer(),
        token_indexers={"tokens": SingleIdTokenIndexer()},
        as_text_field=True,
    )
    instance = reader.text_to_instance(**text_input)
    forward_parameters = list(signature(model.forward).parameters)
    forward_parameters.remove("self")

    assert list(instance.fields.keys()) == forward_parameters
