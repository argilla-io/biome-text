import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.data import Token
from allennlp.data.fields import MetadataField
from allennlp.data.fields import SequenceLabelField
from allennlp.data.fields import TextField
from allennlp.modules import ConditionalRandomField
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.errors import WrongValueError
from biome.text.featurizer import FeaturizeError
from biome.text.helpers import offsets_from_tags
from biome.text.helpers import spacy_to_allennlp_token
from biome.text.helpers import span_labels_to_tag_labels
from biome.text.helpers import tags_from_offsets
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.heads.task_head import TaskHead
from biome.text.modules.heads.task_head import TaskName
from biome.text.modules.heads.task_prediction import Entity
from biome.text.modules.heads.task_prediction import TokenClassificationPrediction


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

    _LOGGER = logging.getLogger(__name__)

    task_name = TaskName.token_classification

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
        if isinstance(text, str):
            doc = self.backbone.tokenizer.nlp(text)
            tokens = [spacy_to_allennlp_token(token) for token in doc]
            tags = (
                tags_from_offsets(doc, entities, self._label_encoding)
                if entities is not None
                else []
            )
            # discard misaligned examples for now
            if "-" in tags:
                self._LOGGER.warning(
                    f"Could not align spans with tokens for following example: '{text}' {entities}"
                )
                return None
        # text is already pre-tokenized
        else:
            tokens = [Token(t) for t in text]

        try:
            instance = self.backbone.featurizer(
                tokens, to_field="text", tokenize=False, aggregate=True
            )
        except FeaturizeError as error:
            self._LOGGER.exception(error)
            return None

        if self.training:
            try:
                instance.add_field(
                    "tags",
                    SequenceLabelField(
                        tags,
                        sequence_field=cast(TextField, instance["text"]),
                        label_namespace=vocabulary.LABELS_NAMESPACE,
                    ),
                )
            except Exception as exception:
                self._LOGGER.exception(str(exception) + f". For `({tokens, tags})`")
                return None

        instance.add_field("raw_text", MetadataField(text))

        return instance

    def forward(  # type: ignore
        self,
        text: TextFieldTensors,
        raw_text: List[Union[str, List[str]]],
        tags: torch.IntTensor = None,
    ) -> Dict:

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

        output = dict(
            viterbi_paths=viterbi_paths,
            raw_text=raw_text,
        )

        if tags is not None:
            output["loss"] = self._loss(logits, tags, mask)
            for metric in self.__all_metrics:
                metric(class_probabilities, tags, mask)

        return output

    def _make_task_prediction(
        self,
        single_forward_output: Dict,
        instance: Instance,
    ) -> TokenClassificationPrediction:
        # The dims are: top_k, tags
        tags: List[List[str]] = self._make_tags(single_forward_output["viterbi_paths"])
        # construct a spacy Doc
        pre_tokenized = not isinstance(single_forward_output["raw_text"], str)
        if pre_tokenized:
            # compose doc from tokens
            doc = Doc(Vocab(), words=single_forward_output["raw_text"])
        else:
            doc = self.backbone.tokenizer.nlp(single_forward_output["raw_text"])

        return TokenClassificationPrediction(
            tags=tags,
            scores=[score for tags, score in single_forward_output["viterbi_paths"]],
            entities=self._make_entities(doc, tags, pre_tokenized),
        )

    def _make_tags(
        self, viterbi_paths: List[Tuple[List[int], float]]
    ) -> List[List[str]]:
        """Makes the 'tags' key of the task prediction"""
        return [
            [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
            for tags, score in viterbi_paths
        ]

    def _make_entities(
        self,
        doc: Doc,
        k_tags: List[List[str]],
        pre_tokenized: bool,
    ) -> List[List[Entity]]:
        """Makes the 'entities' key of the task prediction. Computes offsets with respect to char and token id"""
        return [
            [
                Entity(**entity)
                for entity in offsets_from_tags(
                    doc, tags, self._label_encoding, only_token_spans=pre_tokenized
                )
            ]
            for tags in k_tags
        ]

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
