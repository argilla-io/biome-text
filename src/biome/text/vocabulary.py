"""
    Manages vocabulary tasks and fetches vocabulary information

    Provides utilities for getting information from a given vocabulary.

    Provides management actions such as extending the labels, setting new labels or creating an "empty" vocab.
"""
import logging
from typing import Dict
from typing import List

from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_NON_PADDED_NAMESPACES

from biome.text.features import TransformersFeatures
from biome.text.features import WordFeatures

LABELS_NAMESPACE = "gold_labels"

_LOGGER = logging.getLogger(__name__)


def get_labels(vocab: Vocabulary) -> List[str]:
    """Gets list of labels in the vocabulary

    Parameters
    ----------
    vocab: `allennlp.data.Vocabulary`

    Returns
    -------
    labels: `List[str]`
        A list of label strings
    """
    return [k for k in vocab.get_token_to_index_vocabulary(namespace=LABELS_NAMESPACE)]


def label_for_index(vocab: Vocabulary, idx: int) -> str:
    """Gets label string for a label `int` id

    Parameters
    ----------
    vocab: `allennlp.data.Vocabulary`
    idx: `int
        the token index

    Returns
    -------
    label: `str`
        The string for a label id
    """
    return vocab.get_token_from_index(idx, namespace=LABELS_NAMESPACE)


def index_for_label(vocab: Vocabulary, label: str) -> int:
    """Gets the label `int` id for label string

    Parameters
    ----------
    vocab: `allennlp.data.Vocabulary``
    label: `str`
        the label

    Returns
    -------
    label_idx: `int`
        The label id for label string
    """
    return vocab.get_token_index(label, namespace=LABELS_NAMESPACE)


def get_index_to_labels_dictionary(vocab: Vocabulary) -> Dict[int, str]:
    """Gets a dictionary for turning label `int` ids into label strings

    Parameters
    ----------
    vocab: `allennlp.data.Vocabulary`

    Returns
    -------
    labels: `Dict[int, str]`
        A dictionary to get fetch label strings from ids
    """
    return vocab.get_index_to_token_vocabulary(LABELS_NAMESPACE)


def words_vocab_size(vocab: Vocabulary) -> int:
    """Fetches the vocabulary size for the `words` namespace

    Parameters
    ----------
    vocab: `allennlp.data.Vocabulary`

    Returns
    -------
    size: `int`
        The vocabulary size for the words namespace
    """
    return vocab.get_vocab_size(WordFeatures.namespace)


def extend_labels(vocab: Vocabulary, labels: List[str]):
    """Adds a list of label strings to the vocabulary

    Use this to add new labels to your vocabulary (e.g., useful for reusing the weights of an existing classifier)

    Parameters
    ----------
    vocab: `allennlp.data.Vocabulary`
    labels: `List[str]`
        A list of strings containing the labels to add to an existing vocabulary
    """
    vocab.add_tokens_to_namespace(labels, namespace=LABELS_NAMESPACE)


def set_labels(vocab: Vocabulary, new_labels: List[str]):
    """Resets the labels in the vocabulary with a given labels string list

    Parameters
    ----------
    vocab: `allennlp.data.Vocabulary`
    new_labels: `List[str]`
        The label strings to add to the vocabulary
    """
    for namespace_vocab in [
        vocab.get_token_to_index_vocabulary(LABELS_NAMESPACE),
        vocab.get_index_to_token_vocabulary(LABELS_NAMESPACE),
    ]:
        tokens = list(namespace_vocab.keys())
        for token in tokens:
            del namespace_vocab[token]

    extend_labels(vocab, new_labels)


def create_empty_vocabulary() -> Vocabulary:
    """Creates an empty Vocabulary with configured namespaces

    Returns
    -------
    empty_vocab
        The transformers namespace is added to the `non_padded_namespace`.
    """
    # Following is a hack, because AllenNLP handles the Transformers vocab differently!
    # The transformer vocab has its own padding and oov token, so we add it to the non_padded_namespaces.
    # AllenNLP gives its "transformer vocab" by default the "tags" namespace, which is a non_padded_namespace ...
    # If we do not do this, then writing the vocab to a file and loading it will fail, since AllenNLP will
    # look for its default OVV token in the vocab unless it is flagged as non_padded_namespace.
    # (see the doc string of `allennlp.data.token_indexers.PretrainedTransformerIndexer`)
    return Vocabulary(
        non_padded_namespaces=DEFAULT_NON_PADDED_NAMESPACES
        + (TransformersFeatures.namespace,)
    )


def is_empty(vocab: Vocabulary, namespaces: List[str]) -> bool:
    """Checks if at least one of the given namespaces has an empty vocab.

    Parameters
    ----------
    vocab
        The vocabulary
    namespaces
        Namespaces to check in the vocabulary

    Returns
    -------
    True if one or more namespaces have an empty vocab
    """
    # If a namespace does not exist in the vocab, a default one is created on the fly with a padding and oov token
    # We must drop the padding and out of vocab (oov) tokens -> 2 tokens
    return any([vocab.get_vocab_size(namespace) < 3 for namespace in namespaces])
