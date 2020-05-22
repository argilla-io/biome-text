from typing import Dict

from allennlp.data import TokenIndexer


class InputFeaturizer:
    """Transforms input text (words and/or characters) into indexes and embedding vectors.

    This class defines two input features, words and chars for embeddings at word and character level respectively.

    You can provide additional features by manually specify `indexer` and `embedder` configurations within each
    input feature.

    Parameters
    ----------
    word : `WordFeatures`
        Dictionary defining how to index and embed words
    char : `CharFeatures`
        Dictionary defining how to encode and embed characters
    kwargs :
        Additional params for setting up the features
    """

    def __init__(
        self, indexer: Dict[str, TokenIndexer],
    ):
        self.indexer = indexer
