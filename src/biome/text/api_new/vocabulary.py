from typing import Dict, List

from allennlp.data import Vocabulary

from biome.text.api_new.featurizer import InputFeaturizer


class vocabulary:
    """Extension class for allennlp Vocabulary management"""

    LABELS_NAMESPACE = "gold_labels"

    @classmethod
    def num_classes(cls, vocab: Vocabulary):
        """Number of output classes"""
        return len(cls.get_labels(vocab))

    @classmethod
    def get_labels(cls, vocab: Vocabulary) -> List[str]:
        """The output token classes"""
        return [
            k
            for k in vocab.get_token_to_index_vocabulary(namespace=cls.LABELS_NAMESPACE)
        ]

    @classmethod
    def label_for_index(cls, vocab: Vocabulary, idx: int) -> str:
        """Token label for label index"""
        return vocab.get_token_from_index(idx, namespace=cls.LABELS_NAMESPACE)

    @classmethod
    def index_for_label(cls, vocab: Vocabulary, label: str) -> int:
        """Returns the corresponding label index"""
        return vocab.get_token_index(label, namespace=cls.LABELS_NAMESPACE)

    @classmethod
    def extend_labels(cls, vocab: Vocabulary, labels: List[str]):
        """Extends the number of output labels"""
        vocab.add_tokens_to_namespace(labels, namespace=cls.LABELS_NAMESPACE)

    @classmethod
    def get_index_to_labels_dictionary(cls, vocab: Vocabulary) -> Dict[int, str]:
        """Gets the label ids to label text lookup dictionary"""
        return vocab.get_index_to_token_vocabulary(cls.LABELS_NAMESPACE)

    @classmethod
    def empty_vocab(
        cls, features: InputFeaturizer, labels: List[str] = None
    ) -> Vocabulary:
        """
        This method generate a mock vocabulary for the featurized namespaces.
        If default model use another tokens indexer key name, the pipeline model won't be loaded
        from configuration
        """
        labels = labels or []
        vocab = Vocabulary()
        vocabulary.extend_labels(vocab, labels=labels)
        for namespace in features.feature_keys:
            vocab.add_token_to_namespace("a", namespace=namespace)
        return vocab

    @classmethod
    def set_labels(cls, vocab: Vocabulary, new_labels: List[str]):
        """Reset the labels namespace and set new label list"""
        for namespace_vocab in [
            vocab.get_token_to_index_vocabulary(cls.LABELS_NAMESPACE),
            vocab.get_index_to_token_vocabulary(cls.LABELS_NAMESPACE),
        ]:
            tokens = list(namespace_vocab.keys())
            for token in tokens:
                del namespace_vocab[token]

        cls.extend_labels(vocab, new_labels)

    @classmethod
    def vocab_size(cls, vocab: Vocabulary, namespace: str) -> int:
        """Fetch the namespace vocab size"""
        return vocab.get_vocab_size(namespace=namespace)

    @classmethod
    def words_vocab_size(cls, vocab: Vocabulary) -> int:
        """Returns the words vocabulary size"""
        return cls.vocab_size(vocab, namespace=InputFeaturizer.WORDS)
