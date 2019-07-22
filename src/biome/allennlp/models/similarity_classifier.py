from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, SimilarityFunction
from allennlp.modules.similarity_functions import CosineSimilarity
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("similarity_classifier")
class SimilarityClassifier(Model):
    """
    This ``SimilarityClassifier`` uses a siamese network architecture to perform a binary classification task:
    are two inputs similar or not?
    The two input sequences are encoded with a Seq2VecEncoder, the resulting vectors are concatenated andgg

    simply encodes a sequence of allennlp_2 with a ``Seq2VecEncoder``, then
    predicts a label for the sequence.

    Parameters
    ----------
    vocab
        A Vocabulary, required in order to compute sizes for input/output projections
        and passed on to the :class:`~allennlp.models.model.Model` class.
    text_field_embedder
        Used to embed the input text into a ``TextField``
    seq2seq_encoder
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    dropout
        Dropout percentage to use on the output of the Seq2VecEncoder
    initializer
        Used to initialize the model parameters.
    regularizer
        Used to regularize the model. Passed on to :class:`~allennlp.models.model.Model`.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            dropout: float = None,
            similarity: SimilarityFunction = None,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(
            vocab, regularizer
        )  # Passing on kwargs does not work because of the 'from_params' machinery

        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()*2
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)

        self.similarity = similarity or CosineSimilarity()

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(
            self,  # type: ignore
            record1: Dict[str, torch.LongTensor],
            record2: Dict[str, torch.LongTensor],
            label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        record1
            The first input tokens.
            The dictionary is the output of a ``TextField.as_array()``. It gives names to the tensors created by
            the ``TokenIndexer``s.
            In its most basic form, using a ``SingleIdTokenIndexer``, the dictionary is composed of:
            ``{"tokens": Tensor(batch_size, num_tokens)}``.
            The keys of the dictionary are defined in the `model.yml` input.
            The dictionary is designed to be passed on directly to a ``TextFieldEmbedder``, that has a
            ``TokenEmbedder`` for each key in the dictionary (except you set `allow_unmatched_keys` in the
            ``TextFieldEmbedder`` to False) and knows how to combine different word/character representations into a
            single vector per token in your input.
        record2
            The second input tokens.
        label
            A torch tensor indicating if the two set of input tokens are similar or not (dim: ``(batch_size, 2)``).

        Returns
        -------
        An output dictionary consisting of:
        logits
        class_probabilities
        loss : :class:`~torch.Tensor`, optional
            A scalar loss to be optimised.
        """
        embedded_texts = []
        for tokens in [record1, record2]:
            embedded_text = self._text_field_embedder(tokens)
            mask = get_text_field_mask(tokens).float()

            if self._seq2seq_encoder:
                embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

            embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

            if self._dropout:
                embedded_text = self._dropout(embedded_text)

            embedded_texts.append(embedded_text)

        aggregated_records = torch.cat(embedded_texts, dim=-1)
        logits = self._classification_layer(aggregated_records)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

        # similarity : torch.Tensor = self.similarity(embedded_texts[0], embedded_texts[1])  # dim(batch_size)
        # similarity = similarity.reshape(len(similarity), 1)
        #
        # logits = torch.cat([similarity, 1-similarity], dim=1)  # "fake logits" ...
        #
        # output_dict = {"similarity": similarity}
        #
        # if label is not None:
        #     loss = self._loss(logits, label.long().view(-1))
        #     output_dict["loss"] = loss
        #     self._accuracy(logits, label)
        #
        # return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics

