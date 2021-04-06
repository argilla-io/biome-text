from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import numpy
import torch
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.data.fields import SequenceLabelField
from allennlp.data.fields import TextField
from allennlp.modules import ConditionalRandomField
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure
from allennlp.training.metrics import SpanBasedF1Measure
from torch import Tensor

from biome.text import helpers
from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.featurizer import FeaturizeError
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.heads.task_head import TaskHead
from biome.text.modules.heads.task_prediction import ProfNerPrediction
from biome.text.modules.heads.task_prediction import TaskPrediction


class ProfNer(TaskHead):
    def __init__(
        self,
        backbone: ModelBackbone,
        classification_labels: List[str],
        classification_pooler: Seq2VecEncoderConfiguration,
        ner_tags: List[str],
        ner_tags_encoding: str,
        dropout: float = 0.1,
        ner_feedforward: Optional[FeedForwardConfiguration] = None,
        classification_loss_weight: float = 1.0,
        ner_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(backbone)

        if ner_tags_encoding not in ["BIOUL", "BIO"]:
            raise ValueError(
                f"NER tags encoding '{ner_tags_encoding}' not supported. Allowed values are ['BIOUL', 'BIO']"
            )

        self.backbone.vocab.add_tokens_to_namespace(ner_tags, "ner_tags")
        self.backbone.vocab.add_tokens_to_namespace(
            classification_labels, "classification_labels"
        )

        self._dropout = torch.nn.Dropout(dropout)

        self._classification_pooler = classification_pooler.input_dim(
            self.backbone.encoder.get_output_dim()
        ).compile()
        self._classification_layer = torch.nn.Linear(
            self._classification_pooler.get_output_dim(), len(classification_labels)
        )

        self._classification_loss = torch.nn.CrossEntropyLoss()
        self._classification_loss_weight = classification_loss_weight

        self._ner_feedforward: Optional[FeedForward] = (
            None
            if ner_feedforward is None
            else ner_feedforward.input_dim(
                self.backbone.encoder.get_output_dim()
            ).compile()
        )
        self._encoding_output_dim = (
            self.backbone.encoder.get_output_dim()
            if self._ner_feedforward is None
            else self._ner_feedforward.get_output_dim()
        )
        self._tag_layer = TimeDistributed(
            torch.nn.Linear(self._encoding_output_dim, len(ner_tags))
        )

        constraints = allowed_transitions(
            ner_tags_encoding,
            self.backbone.vocab.get_index_to_token_vocabulary("ner_tags"),
        )
        self._crf = ConditionalRandomField(len(ner_tags), constraints)
        self._ner_loss_weight = ner_loss_weight

        self.metrics = {
            "classification_accuracy": CategoricalAccuracy(),
            "classification_micro": FBetaMeasure(average="micro"),
            "classification_macro": FBetaMeasure(average="macro"),
            "classification_label": FBetaMeasure(
                labels=list(range(len(classification_labels)))
            ),
            "ner_f1": SpanBasedF1Measure(
                self.backbone.vocab, "ner_tags", label_encoding=ner_tags_encoding
            ),
            "valid_classification_accuracy": CategoricalAccuracy(),
            "valid_classification_micro": FBetaMeasure(average="micro"),
            "valid_classification_macro": FBetaMeasure(average="macro"),
            "valid_classification_label": FBetaMeasure(
                labels=list(range(len(classification_labels)))
            ),
            "valid_ner_f1": SpanBasedF1Measure(
                self.backbone.vocab, "ner_tags", label_encoding=ner_tags_encoding
            ),
        }

    def featurize(
        self,
        tokens: List[str],
        labels: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Instance:
        """

        Parameters
        ----------
        tokens
            spacy tokenized input
        labels
            The classification label
        tags
            The NER tags

        Returns
        -------
        instance

        Raises
        ------
        FeaturizeError
        """
        if not isinstance(tokens, list):
            raise FeaturizeError("Input argument 'tokens' has to be a list of strings")
        if labels is not None and tags is None:
            raise FeaturizeError("You are missing the tags!")
        if labels is None and tags is not None:
            raise FeaturizeError("You are missing the labels!")

        instance = self.backbone.featurizer(
            tokens, to_field="tokens", aggregate=True, tokenize=False
        )

        if labels is not None and tags is not None:
            if len(tags) != len(tokens):
                raise FeaturizeError(
                    f"The number of tags does not match the number of tokens! {tokens} {tags}"
                )
            label_field = LabelField(labels, label_namespace="classification_labels")
            instance.add_field("labels", label_field)

            try:
                sequence_label_field = SequenceLabelField(
                    tags,
                    sequence_field=cast(TextField, instance["tokens"]),
                    label_namespace="ner_tags",
                )
            except Exception as error:
                raise FeaturizeError(
                    f"Could not create SequenceLabelField for: {tokens, tags}"
                ) from error
            else:
                instance.add_field("tags", sequence_label_field)

        return instance

    def forward(
        self,
        tokens: Dict[str, Union[Tensor, Dict[str, Tensor]]],
        labels: Tensor = None,
        tags: Tensor = None,
    ):
        """

        Parameters
        ----------
        tokens
            Word tokens produced by the spacy tokenizer
        labels
            Classification label
        tags
            NER tags for the word tokens

        Returns
        -------

        """
        mask = get_text_field_mask(tokens)
        embedded_tokens = self._dropout(self.backbone(tokens, mask))

        classification_logits = self._classification_layer(
            self._classification_pooler(embedded_tokens, mask)
        )

        if self._ner_feedforward is not None:
            embedded_tokens = self._ner_feedforward(embedded_tokens)
        ner_logits = self._tag_layer(embedded_tokens)

        viterbi_paths: List[Tuple[List[int], float]] = self._crf.viterbi_tags(
            ner_logits, mask
        )

        output = dict(
            classification_logits=classification_logits, viterbi_paths=viterbi_paths
        )

        if labels is not None and tags is not None:
            # Classification loss
            output["loss"] = (
                self._classification_loss(classification_logits, labels)
                * self._classification_loss_weight
            )

            # NER loss
            output["loss"] += (
                -1 * self._crf(ner_logits, tags, mask) * self._ner_loss_weight
            )

            ner_logits_for_metrics = torch.zeros_like(ner_logits)
            for batch_id, instance_tags in enumerate(viterbi_paths):
                for token_id, tag_id in enumerate(instance_tags[0]):
                    ner_logits_for_metrics[batch_id, token_id, tag_id] = 1

            # metrics
            if self.training:
                self.metrics["classification_accuracy"](classification_logits, labels)
                self.metrics["classification_micro"](classification_logits, labels)
                self.metrics["classification_macro"](classification_logits, labels)
                self.metrics["classification_label"](classification_logits, labels)
                self.metrics["ner_f1"](ner_logits_for_metrics, tags, mask)
            else:
                self.metrics["valid_classification_accuracy"](
                    classification_logits, labels
                )
                self.metrics["valid_classification_micro"](
                    classification_logits, labels
                )
                self.metrics["valid_classification_macro"](
                    classification_logits, labels
                )
                self.metrics["valid_classification_label"](
                    classification_logits, labels
                )
                self.metrics["valid_ner_f1"](ner_logits_for_metrics, tags, mask)

        return output

    def _make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
    ) -> TaskPrediction:

        # softmax is not implemented in numpy ...
        classification_logits = torch.from_numpy(
            single_forward_output["classification_logits"]
        )
        classification_probabilities = (
            torch.nn.functional.softmax(classification_logits, dim=0)
            .sort(descending=True)[0]
            .tolist()
        )
        classification_labels = [
            self.backbone.vocab.get_token_from_index(int(ind), "classification_labels")
            for ind in classification_logits.argsort(descending=True)
        ]
        ner_tags = [
            self.backbone.vocab.get_token_from_index(ind, "ner_tags")
            for ind in single_forward_output["viterbi_paths"][0]
        ]

        return ProfNerPrediction(
            classification_labels=classification_labels,
            classification_probabilities=classification_probabilities,
            ner_tags=ner_tags,
        )

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.training:
            metrics = {
                "classification/accuracy": self.metrics[
                    "classification_accuracy"
                ].get_metric(reset)
            }
            for key, value in (
                self.metrics["classification_micro"].get_metric(reset).items()
            ):
                metrics.update({f"classification/micro_{key}": value})
            for key, value in (
                self.metrics["classification_macro"].get_metric(reset).items()
            ):
                metrics.update({f"classification/macro_{key}": value})
            for key, values in (
                self.metrics["classification_label"].get_metric(reset).items()
            ):
                for i, v in enumerate(values):
                    label = vocabulary.label_for_index(self.backbone.vocab, i)
                    # sanitize label using same patterns as tensorboardX to avoid summary writer warnings
                    label = helpers.sanitize_metric_name(label)
                    metrics.update({"_{}/{}".format(key, label): v})
            for key, value in self.metrics["ner_f1"].get_metric(reset).items():
                metrics.update({f"ner/{key}": value})
        else:
            metrics = {
                "valid_classification/accuracy": self.metrics[
                    "valid_classification_accuracy"
                ].get_metric(reset)
            }
            for key, value in (
                self.metrics["valid_classification_micro"].get_metric(reset).items()
            ):
                metrics.update({f"valid_classification/micro_{key}": value})
            for key, value in (
                self.metrics["valid_classification_macro"].get_metric(reset).items()
            ):
                metrics.update({f"valid_classification/macro_{key}": value})
            for key, values in (
                self.metrics["valid_classification_label"].get_metric(reset).items()
            ):
                for i, v in enumerate(values):
                    label = vocabulary.label_for_index(self.backbone.vocab, i)
                    # sanitize label using same patterns as tensorboardX to avoid summary writer warnings
                    label = helpers.sanitize_metric_name(label)
                    metrics.update({"valid_{}/{}".format(key, label): v})
            for key, value in self.metrics["valid_ner_f1"].get_metric(reset).items():
                metrics.update({f"valid_ner/{key}": value})

        return metrics


class ProfNerConfiguration(ComponentConfiguration[ProfNer]):
    """Configuration for classification head components"""

    pass
