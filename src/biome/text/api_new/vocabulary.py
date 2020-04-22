from typing import Dict, List

from allennlp.data import Vocabulary

from biome.text.api_new.featurizer import InputFeaturizer


class vocabulary:
    """Manages vocabulary tasks and fetches vocabulary information
    
    Provides utilities for getting information from a given vocabulary.
    
    Provides management actions such as extending the labels, setting new labels or creating an "empty" vocab.
    """

    """Namespace for labels in the vocabulary"""
    LABELS_NAMESPACE = "gold_labels"

    @classmethod
    def num_labels(cls, vocab: Vocabulary) -> int:
        """Gives the number of labels in the vocabulary
        
        # Parameters
            vocab: `allennlp.data.Vocabulary`

        # Returns
            The number of classes in the label vocabulary
        """
        return len(cls.get_labels(vocab))

    @classmethod
    def get_labels(cls, vocab: Vocabulary) -> List[str]:
        """Gets list of labels in the vocabulary
        
        # Parameters
            vocab: `allennlp.data.Vocabulary`

        # Returns
            A `List` of label `str`
        """
        return [
            k
            for k in vocab.get_token_to_index_vocabulary(namespace=cls.LABELS_NAMESPACE)
        ]

    @classmethod
    def label_for_index(cls, vocab: Vocabulary, idx: int) -> str:
        """Gets label string for a label `int` id
        
        # Parameters
            vocab: `allennlp.data.Vocabulary`

        # Returns
            The `str` for a label id
        """
        return vocab.get_token_from_index(idx, namespace=cls.LABELS_NAMESPACE)

    @classmethod
    def index_for_label(cls, vocab: Vocabulary, label: str) -> int:
        """Gets the label `int` id for label string
        
        # Parameters
            vocab: `allennlp.data.Vocabulary`

        # Returns
            The `int` id for a label string
        """
        return vocab.get_token_index(label, namespace=cls.LABELS_NAMESPACE)

    @classmethod
    def get_index_to_labels_dictionary(cls, vocab: Vocabulary) -> Dict[int, str]:
        """Gets a dictionary for turning label `int` ids into label strings

        # Parameters
            vocab: `allennlp.data.Vocabulary`

        # Returns
            A  dictionary `Dict[int, str]` to get labels strings from ids
        """
        return vocab.get_index_to_token_vocabulary(cls.LABELS_NAMESPACE)

    @classmethod
    def vocab_size(cls, vocab: Vocabulary, namespace: str) -> int:
        """Fetches the vocabulary size of a given namespace

        # Parameters
            vocab: `allennlp.data.Vocabulary`
            namespace: `str`
        """
        return vocab.get_vocab_size(namespace=namespace)

    @classmethod
    def words_vocab_size(cls, vocab: Vocabulary) -> int:
        """Fetches the vocabulary size for the `words` namespace

        # Parameters
            vocab: `allennlp.data.Vocabulary`
        """
        return cls.vocab_size(vocab, namespace=InputFeaturizer.WORDS)

    @classmethod
    def extend_labels(cls, vocab: Vocabulary, labels: List[str]):
        """Adds a list of label strings to the vocabulary
        
        Use this to add new labels to your vocabulary (e.g., useful for reusing the weights of an existing classifier)
        
        # Parameters
            vocab: `allennlp.data.Vocabulary`
            labels: `List[str]`
                A list of strings containing the labels to add to an existing vocabulary
        """
        vocab.add_tokens_to_namespace(labels, namespace=cls.LABELS_NAMESPACE)

    @classmethod
    def empty_vocab(
        cls, featurizer: InputFeaturizer, labels: List[str] = None
    ) -> Vocabulary:
        """Generates a "mock" empty vocabulary for a given `InputFeaturizer`
        
        This method generate a mock vocabulary for the featurized namespaces.
        TODO: Clarify? --> If default model use another tokens indexer key name, the pipeline model won't be loaded from configuration
        # Parameters
            featurizer: `InputFeaturizer`
                A featurizer for which to create the vocabulary
            labels: `List[str]`
                The label strings to add to the vocabulary
        # Returns
            The created `allennlp.data.Vocabulary`
        """
        labels = labels or []
        vocab = Vocabulary()
        vocabulary.extend_labels(vocab, labels=labels)
        for namespace in featurizer.feature_keys:
            vocab.add_token_to_namespace("a", namespace=namespace)
        return vocab

    @classmethod
    def set_labels(cls, vocab: Vocabulary, new_labels: List[str]):
        """Resets the labels in the vocabulary with a given labels string list

        # Parameters
            vocab: `allennlp.data.Vocabulary`
            new_labels: `List[str]`
                The label strings to add to the vocabulary
        """
        for namespace_vocab in [
            vocab.get_token_to_index_vocabulary(cls.LABELS_NAMESPACE),
            vocab.get_index_to_token_vocabulary(cls.LABELS_NAMESPACE),
        ]:
            tokens = list(namespace_vocab.keys())
            for token in tokens:
                del namespace_vocab[token]

        cls.extend_labels(vocab, new_labels)
