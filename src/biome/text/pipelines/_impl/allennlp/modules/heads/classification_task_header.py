from typing import Type, Optional, Dict, List

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import (
    TimeDistributed,
    TextFieldEmbedder,
    Seq2VecEncoder,
    Seq2SeqEncoder,
    FeedForward,
)
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from torch.nn import Linear

from biome.text.pipelines._impl.allennlp.modules.heads.task_head import TaskHead
from biome.text.pipelines._impl.allennlp.modules.layer_spec import (
    Seq2VecEncoderSpec,
    Seq2SeqEncoderSpec,
    FeedForwardSpec,
)


class ClassificationHead(TaskHead):

    __LABELS_NAMESPACE = "labels"

    def __init__(
        self,
        vocab: Vocabulary,
        prev_layer: Seq2SeqEncoder,
        pooler: Seq2VecEncoderSpec,
        tokens_pooler: Optional[Seq2VecEncoderSpec] = None,
        encoder: Optional[Seq2SeqEncoderSpec] = None,
        feedforward: Optional[FeedForwardSpec] = None,
        multilabel: bool = False,
    ) -> None:
        super(ClassificationHead, self).__init__(vocab)

        # layers
        tokens_pooler = (
            BagOfEmbeddingsEncoder(embedding_dim=prev_layer.get_output_dim())
            if not tokens_pooler
            else tokens_pooler.input_dim(prev_layer.get_output_dim()).compile()
        )

        self._tokens_pooler = TimeDistributed(tokens_pooler)
        self._doc_encoder = (
            PassThroughEncoder(tokens_pooler.get_output_dim())
            if not encoder
            else encoder.input_dim(tokens_pooler.get_output_dim()).compile()
        )
        self._doc_pooler = pooler.input_dim(
            self._doc_encoder.get_output_dim()
        ).compile()
        self._feedforward = (
            None
            if not feedforward
            else feedforward.input_dim(self._doc_pooler.get_output_dim()).compile()
        )

        # output layers
        self._classifier_input_dim = (
            self._feedforward.get_output_dim()
            if self._feedforward
            else self._doc_pooler.get_output_dim()
        )

        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self.num_classes
        )

        # label related configurations
        self._multilabel = multilabel
        self._calculate_output = (
            self._output_multilabel if self._multilabel else self._output_label
        )

        # metrics and loss
        if self._multilabel:
            self._metrics = None  # CategoricalAccuracy() TODO: for multilabel we want to calculate F1 per label and/or ROC-AUC
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            # metrics, some AllenNLP models use the names _accuracy or _metrics, so we have to be more specific.
            self._metrics = {"accuracy": CategoricalAccuracy()}
            self._metrics.update(
                {
                    label: F1Measure(index)
                    for index, label in enumerate(self.output_classes)
                }
            )
            self._loss = torch.nn.CrossEntropyLoss()

    @property
    def num_classes(self):
        """Number of output classes"""
        return len(self.output_classes)

    @property
    def output_classes(self) -> List[str]:
        """The output token classes"""
        return [
            k
            for k in self.vocab.get_token_to_index_vocabulary(
                namespace=self.__LABELS_NAMESPACE
            )
        ]

    def label_for_index(self, idx) -> str:
        """Token label for label index"""
        return self.vocab.get_token_from_index(idx, namespace=self.__LABELS_NAMESPACE)

    def extend_labels(self, labels: List[str]):
        """Extends the number of output labels"""
        self.vocab.add_tokens_to_namespace(labels, namespace=self.__LABELS_NAMESPACE)
        self.self._classification_layer = Linear(
            self._classifier_input_dim, self.num_classes
        )

    def forward(  # type: ignore
        self,
        embedded_text: torch.Tensor,
        mask: torch.BoolTensor,
        label: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        embedded_text : `torch.Tensor`, required.
            A tensor of shape (batch_size, num_sentences, timesteps, output_dim)
        # Returns
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        embedded_text = self._tokens_pooler(embedded_text, mask=mask)
        encoded_text_mask = get_text_field_mask({"encoded_text": embedded_text})
        embedded_text = self._doc_encoder(embedded_text, mask=encoded_text_mask)
        embedded_text = self._doc_pooler(embedded_text, mask=encoded_text_mask)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return self._calculate_output(label, logits)

    def _output_label(
        self, label: Optional[torch.IntTensor], logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output_dict = {"logits": logits}
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict["probs"] = probs
        if label is not None:
            loss = self._loss(logits, label.long())
            output_dict["loss"] = loss
            for metric in self._metrics.values():
                metric(logits, label)

        return output_dict

    def _output_multilabel(
        self, label: Optional[torch.IntTensor], logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output_dict = {"logits": logits}
        probs = logits.sigmoid()  # TODO check this 'probs' calculation
        output_dict["probs"] = probs
        if label is not None:
            # casting long to float for BCELoss
            # see https://discuss.pytorch.org/t/nn-bcewithlogitsloss-cant-accept-one-hot-target/59980
            loss = self._loss(
                logits.view(-1, self.num_classes),
                label.view(-1, self.num_classes).type_as(logits),
            )
            output_dict["loss"] = loss
            # self._metrics(logits, label) TODO
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Get the metrics of our classifier, see :func:`~allennlp_2.models.Model.get_metrics`.

        Parameters
        ----------
        reset
            Reset the metrics after obtaining them?

        Returns
        -------
        A dictionary with all metric names and values.
        """
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self._metrics.items():
            if metric_name == "accuracy":
                all_metrics["accuracy"] = metric.get_metric(reset)
            else:
                # pylint: disable=invalid-name
                precision, recall, f_1 = metric.get_metric(
                    reset
                )  # pylint: disable=invalid-name
                total_f1 += f_1
                total_precision += precision
                total_recall += recall
                all_metrics[metric_name + "/f1"] = f_1
                all_metrics[metric_name + "/precision"] = precision
                all_metrics[metric_name + "/recall"] = recall

        num_classes = self.num_classes
        all_metrics["average/f1"] = total_f1 / num_classes
        all_metrics["average/precision"] = total_precision / num_classes
        all_metrics["average/recall"] = total_recall / num_classes

        return all_metrics
