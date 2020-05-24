"""
    Manages vocabulary tasks and fetches vocabulary information

    Provides utilities for getting information from a given vocabulary.

    Provides management actions such as extending the labels, setting new labels or creating an "empty" vocab.
"""
from typing import Dict, List

from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from biome.text.featurizer import WordFeatures

LABELS_NAMESPACE = "gold_labels"


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


def empty_vocabulary(namespaces):
    """
    Creates an empty vocabulary. Used for early pipeline initialization

    Arguments
    ----------
    namespaces: `List[str]`
        The vocab namespaces to create
    """
    vocab = Vocabulary()
    for namespace in namespaces:
        vocab.add_token_to_namespace(DEFAULT_OOV_TOKEN, namespace=namespace)
    return vocab


def is_empty(vocab: Vocabulary, namespaces: List[str]) -> bool:
    """
    Checks if a vocab is empty respect to given namespaces

    Returns True vocab size is 0 for all given namespaces
    """
    for namespaces in namespaces:
        # We must drop the padding and out of vocab tokens = 2 tokens
        if vocab.get_vocab_size(namespaces) > 2:
            return False
    return True
