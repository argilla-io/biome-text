import pytest
from allennlp.data import Instance
from allennlp.data.fields import ListField
from allennlp.data.fields import TextField
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab

from biome.text.configuration import PredictionConfiguration
from biome.text.helpers import spacy_to_allennlp_token
from biome.text.modules.heads import TaskHead
from biome.text.modules.heads import TaskPrediction
from biome.text.modules.heads.task_prediction import Token


@pytest.fixture
def task_head() -> TaskHead:
    return TaskHead(backbone="mock_backbone")


def test_prediction_not_implemented(task_head):
    with pytest.raises(NotImplementedError):
        task_head.make_task_prediction("mock", "mock", "mock")


def test_attributions_not_implemented(task_head, monkeypatch):
    def mock_make_task_prediction(*args, **kwargs):
        return TaskPrediction()

    monkeypatch.setattr(task_head, "_make_task_prediction", mock_make_task_prediction)

    with pytest.raises(NotImplementedError):
        task_head.make_task_prediction(
            "mock", "mock", PredictionConfiguration(add_attributions=True)
        )


def test_make_task_prediction(monkeypatch, task_head):
    def mock_make_task_prediction(*args, **kwargs):
        return TaskPrediction()

    def mock_compute_attributions(*args, **kwargs):
        return kwargs

    def mock_extract_tokens(*args, **kwargs):
        return "tokens"

    monkeypatch.setattr(task_head, "_make_task_prediction", mock_make_task_prediction)
    monkeypatch.setattr(task_head, "_compute_attributions", mock_compute_attributions)
    monkeypatch.setattr(task_head, "_extract_tokens", mock_extract_tokens)

    prediction = task_head.make_task_prediction(
        "mock_forward_output",
        "mock_instance",
        PredictionConfiguration(
            add_tokens=True,
            add_attributions=True,
            attributions_kwargs={"test": "kwarg"},
        ),
    )

    assert isinstance(prediction, TaskPrediction)
    assert hasattr(prediction, "tokens") and hasattr(prediction, "attributions")
    assert prediction.tokens == "tokens"
    assert prediction.attributions == {"test": "kwarg"}


@pytest.mark.parametrize("allennlp_tokens", [False, True])
def test_extract_tokens(task_head, allennlp_tokens):
    tokenizer = Tokenizer(Vocab())
    input_tokens = list(tokenizer("test this sentence."))
    if allennlp_tokens:
        input_tokens = [spacy_to_allennlp_token(tok) for tok in input_tokens]

    tf = TextField(input_tokens, None)
    instance = Instance({"test": tf})

    tokens = task_head._extract_tokens(instance)

    assert all([isinstance(tok, Token) for tok in tokens])
    assert all(itok.text == otok.text for itok, otok in zip(input_tokens, tokens))
    assert all(itok.idx == otok.start for itok, otok in zip(input_tokens, tokens))
    if allennlp_tokens:
        assert all(itok.idx_end == otok.end for itok, otok in zip(input_tokens, tokens))
    else:
        assert all(
            itok.idx + len(itok.text) == otok.end
            for itok, otok in zip(input_tokens, tokens)
        )
    assert all([tok.field == "test" for tok in tokens])


def test_extract_tokens_listfield(task_head):
    tokenizer = Tokenizer(Vocab())
    input_tokens = list(tokenizer("test this sentence."))

    tf = TextField(input_tokens, None)
    instance = Instance({"test": ListField([tf, tf])})

    tokens = task_head._extract_tokens(instance)

    assert len(tokens) == 2 and len(tokens[0]) == 3 and len(tokens[1]) == 3
    assert all(
        [all([isinstance(tok, Token) for tok in tf_tokens] for tf_tokens in tokens)]
    )
