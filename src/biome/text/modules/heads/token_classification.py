import logging
from typing import Dict, List, Optional, Union, cast, Tuple

import torch
from allennlp.data import Instance, TextFieldTensors
from allennlp.data.fields import SequenceLabelField, TextField, MetadataField
from allennlp.modules import ConditionalRandomField, FeedForward, TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from spacy.tokens.doc import Doc

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import (
    ComponentConfiguration,
    FeedForwardConfiguration,
)
from biome.text.helpers import (
    span_labels_to_tag_labels,
    tags_from_offsets,
    offsets_from_tags,
)
from .task_head import TaskHead, TaskName, TaskOutput
from ...errors import WrongValueError


class TokenClassification(TaskHead):
    """Task head for token classification (NER)

    Parameters
    ----------
    backbone
        The model backbone
    labels
        List span labels. Span labels get converted to tag labels internally, using
        configured label_encoding for that.
    label_encoding
        The format of the tags. Supported encodings are: ['BIO', 'BIOUL']
    top_k
    dropout
    feedforward
    """

    __LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        label_encoding: Optional[str] = "BIOUL",
        top_k: int = 1,
        dropout: Optional[float] = 0.0,
        feedforward: Optional[FeedForwardConfiguration] = None,
    ) -> None:
        super(TokenClassification, self).__init__(backbone)

        if label_encoding not in ["BIOUL", "BIO"]:
            raise WrongValueError(
                f"Label encoding {label_encoding} not supported. Allowed values are {['BIOUL', 'BIO']}"
            )

        self._span_labels = labels
        self._label_encoding = label_encoding

        vocabulary.set_labels(
            self.backbone.vocab,
            # Convert span labels to tag labels if necessary
            # We just check if "O" is in the label list, a necessary tag for IOB/BIOUL schemes,
            # an unlikely label for spans
            span_labels_to_tag_labels(labels, self._label_encoding),
        )

        self.top_k = top_k
        self.dropout = torch.nn.Dropout(dropout)
        self._feedforward: FeedForward = (
            None
            if not feedforward
            else feedforward.input_dim(backbone.encoder.get_output_dim()).compile()
        )
        # output layers
        self._classifier_input_dim = (
            self._feedforward.get_output_dim()
            if self._feedforward
            else backbone.encoder.get_output_dim()
        )
        # we want this linear applied to each token in the sequence
        self._label_projection_layer = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        )
        constraints = allowed_transitions(
            self._label_encoding,
            vocabulary.get_index_to_labels_dictionary(self.backbone.vocab),
        )

        self._crf = ConditionalRandomField(
            self.num_labels, constraints, include_start_end_transitions=True
        )

        self.metrics = {"accuracy": CategoricalAccuracy()}
        if self.top_k > 1:
            self.metrics.update(
                {f"accuracy_{self.top_k}": CategoricalAccuracy(top_k=self.top_k)}
            )
        self.f1_metric = SpanBasedF1Measure(
            self.backbone.vocab,
            tag_namespace=vocabulary.LABELS_NAMESPACE,
            label_encoding=self._label_encoding,
        )

        self.__all_metrics = [self.f1_metric]
        self.__all_metrics.extend(self.metrics.values())

    @property
    def span_labels(self) -> List[str]:
        return self._span_labels

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """loss is calculated as -log_likelihood from crf"""
        return -1 * self._crf(logits, labels, mask)

    def featurize(
        self,
        text: Union[str, List[str]],
        entities: Optional[List[dict]] = None,
        tags: Optional[Union[List[str], List[int]]] = None,
    ) -> Optional[Instance]:
        """
        Parameters
        ----------
        text
            Can be either a simple str or a list of str,
            in which case it will be treated as a list of pretokenized tokens
        entities
            A list of span labels

            Span labels are dictionaries that contain:

            'start': int, char index of the start of the span
            'end': int, char index of the end of the span (exclusive)
            'label': str, label of the span

            They are used with the `spacy.gold.biluo_tags_from_offsets` method.
        tags
            A list of tags in the BIOUL or BIO format.
        """

        pre_tokenized = not isinstance(text, str)
        if pre_tokenized:  # compose spacy doc from tokens
            doc = Doc(self.backbone.tokenizer.nlp.vocab, words=text)
        else:
            doc = self.backbone.tokenizer.nlp(text)
            tags = (
                tags_from_offsets(doc, entities, self._label_encoding)
                if entities is not None
                else []
            )
            # discard misaligned examples for now
            if "-" in tags:
                self.__LOGGER.warning(
                    f"Could not align spans with tokens for following example: '{text}' {entities}"
                )
                return None

        instance = self._featurize_tokens([token.text for token in doc], tags)
        instance.add_field("spacy_doc", MetadataField(doc))
        instance.add_field("pre_tokenized", MetadataField(pre_tokenized))

        return instance

    def _featurize_tokens(
        self, tokens: List[str], tags: Union[List[str], List[int]]
    ) -> Optional[Instance]:
        """Create an example Instance from token and tags"""

        instance = self.backbone.featurizer(
            tokens, to_field="text", tokenize=False, aggregate=True
        )

        if self.training:
            assert tags, f"No tags found when training. Data [{tokens, tags}]"
            instance.add_field(
                "tags",
                SequenceLabelField(
                    tags,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace=vocabulary.LABELS_NAMESPACE,
                ),
            )
        return instance

    def task_name(self) -> TaskName:
        return TaskName.token_classification

    def forward(  # type: ignore
        self,
        text: TextFieldTensors,
        spacy_doc: List[Doc],
        pre_tokenized: List[bool],
        tags: torch.IntTensor = None,
    ) -> TaskOutput:

        mask = get_text_field_mask(text)
        embedded_text = self.dropout(self.backbone.forward(text, mask))

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._label_projection_layer(embedded_text)
        # dims are: batch, top_k, (tag_sequence, viterbi_score)
        viterbi_paths: List[List[Tuple[List[int], float]]] = self._crf.viterbi_tags(
            logits, mask, top_k=self.top_k
        )
        # We just keep the best path for every instance
        predicted_tags: List[List[int]] = [paths[0][0] for paths in viterbi_paths]
        class_probabilities = logits * 0.0

        for i, instance_tags in enumerate(predicted_tags):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities[i, j, tag_id] = 1

        output = TaskOutput(
            logits=logits,
            probs=class_probabilities,
            viterbi_paths=viterbi_paths,
            mask=mask,
            tokenization=(spacy_doc, pre_tokenized),
        )

        if tags is not None:
            output.loss = self._loss(logits, tags, mask)
            for metric in self.__all_metrics:
                metric(class_probabilities, tags, mask)

        return output

    def _decode_tags(self, k_tags: List[List[int]]) -> List[List[str]]:
        """Decode predicted tags"""
        return [
            [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
            for tags, prob in k_tags
        ]

    def _decode_entities(
        self, doc: Doc, tags: List[str], pre_tokenized: bool
    ) -> List[Dict]:
        """Decode predicted entities from tags"""
        return offsets_from_tags(
            doc, tags, self._label_encoding, only_token_spans=pre_tokenized
        )

    def _decode_tokens(self, doc: Doc) -> List[Dict]:
        """Decode tokens"""
        return [
            {"text": token.text, "start": token.idx, "end": token.idx + len(token)}
            for token in doc
        ]

    def decode(self, output: TaskOutput) -> TaskOutput:
        # Te dims are: batch, k_tags, tags
        output.tags: List[List[List[str]]] = [
            self._decode_tags(k_tags) for k_tags in output.viterbi_paths
        ]

        # fmt: off
        output.entities, output.tokens = zip(*[
            (
                [self._decode_entities(doc, tags, pre_tokenized) for tags in k_tags],
                self._decode_tokens(doc) if not pre_tokenized else None
            )
            for doc, pre_tokenized, k_tags in zip(*output.tokenization, output.tags)
        ])
        # fmt: on

        output.tokens = [tokens for tokens in output.tokens if tokens]
        if len(output.tokens) < 1:  # drop tokens field if no data
            del output.tokens

        del output.logits
        del output.mask
        del output.probs
        del output.tokenization
        del output.viterbi_paths

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
            if metric_name != "f1"
        }
        metrics.update(self.f1_metric.get_metric(reset=reset))
        return metrics


class TokenClassificationConfiguration(ComponentConfiguration[TokenClassification]):
    """Configuration for classification head components"""

    pass
