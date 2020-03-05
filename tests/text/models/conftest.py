import pytest
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary


@pytest.fixture
def tokens_labels_vocab():
    text = "The grass is green and the sky is blue. What can you do?"
    text_field = TextField(
        WordTokenizer().tokenize(text),
        token_indexers={"tokens": SingleIdTokenIndexer()},
    )

    vocab = Vocabulary.from_instances(
        [text_field, LabelField("label0"), LabelField("label1")]
    )

    return vocab
