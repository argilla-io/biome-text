from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy
import numpy as np
import torch
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data.fields import ArrayField
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
from torch import BoolTensor
from torch import Tensor
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast

from biome.text.backbone import ModelBackbone
from biome.text.featurizer import FeaturizeError
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.heads.task_head import TaskHead
from biome.text.modules.heads.task_prediction import ProfNerPrediction
from biome.text.modules.heads.task_prediction import TaskPrediction


class ProfNerT(TaskHead):
    def __init__(
        self,
        backbone: ModelBackbone,
        classification_labels: List[str],
        classification_pooler: Seq2VecEncoderConfiguration,
        ner_tags: List[str],
        ner_tags_encoding: str,
        transformers_model: str,
        dropout: float = 0.1,
        feedforward: Optional[FeedForwardConfiguration] = None,
    ) -> None:
        super().__init__(backbone)

        if ner_tags_encoding not in ["BIOUL", "BIO"]:
            raise ValueError(
                f"NER tags encoding '{ner_tags_encoding}' not supported. Allowed values are ['BIOUL', 'BIO']"
            )

        self._transformer_tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast
        ] = AutoTokenizer.from_pretrained(transformers_model)

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

        self._feedforward: Optional[FeedForward] = (
            None
            if feedforward is None
            else feedforward.input_dim(self.backbone.encoder.get_output_dim()).compile()
        )
        self._encoding_output_dim = (
            self.backbone.encoder.get_output_dim()
            if self._feedforward is None
            else self._feedforward.get_output_dim()
        )
        self._tag_layer = TimeDistributed(
            torch.nn.Linear(self._encoding_output_dim, len(ner_tags))
        )

        constraints = allowed_transitions(
            ner_tags_encoding,
            self.backbone.vocab.get_index_to_token_vocabulary("ner_tags"),
        )
        self._crf = ConditionalRandomField(len(ner_tags), constraints)

        self.metrics = {
            "classification_accuracy": CategoricalAccuracy(),
            "classification_micro": FBetaMeasure(average="micro"),
            "classification_macro": FBetaMeasure(average="macro"),
            "ner_f1": SpanBasedF1Measure(
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
        if (labels is not None and tags is None) or (
            labels is None and tags is not None
        ):
            raise FeaturizeError("You are missing either labels or tags!")

        input_ids = self._transformer_tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        transformer_tokens_str = self._transformer_tokenizer.convert_ids_to_tokens(
            input_ids["input_ids"]
        )
        # We only want to tag the first word piece of the word tokens
        word_piece_mask = np.array(input_ids["offset_mapping"], dtype=int)[:, 0] == 0
        ner_tokens_mask = word_piece_mask & ~np.array(
            input_ids["special_tokens_mask"], dtype=bool
        )

        instance = self.backbone.featurizer(
            transformer_tokens_str, to_field="subtokens", aggregate=True, tokenize=False
        )

        array_field = ArrayField(ner_tokens_mask, padding_value=0, dtype=bool)
        instance.add_field("ner_subtokens_mask", array_field)

        if labels is not None and tags is not None:
            label_field = LabelField(labels, label_namespace="classification_labels")
            instance.add_field("labels", label_field)

            text_field = TextField(
                [Token(token) for token in tokens],
                token_indexers={},
            )
            try:
                sequence_label_field = SequenceLabelField(
                    tags,
                    sequence_field=text_field,
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
        subtokens: Dict[str, Union[Tensor, Dict[str, Tensor]]],
        ner_subtokens_mask: BoolTensor,
        tokens: Dict[str, Union[Tensor, Dict[str, Tensor]]] = None,
        labels: Tensor = None,
        tags: Tensor = None,
    ):
        """

        Parameters
        ----------
        subtokens
            Subword tokens produced by the transformers tokenizer
        ner_subtokens_mask
            Masks the first subtoken belonging to a word token
        tokens
            Word tokens produced by the spacy tokenizer
        labels
            Classification label
        tags
            NER tags for the word tokens

        Returns
        -------

        """
        # This is the mask for padded word pieces
        mask = get_text_field_mask(subtokens)
        embedded_tokens = self._dropout(self.backbone(subtokens, mask))

        classification_logits = self._classification_layer(
            self._classification_pooler(embedded_tokens, mask)
        )

        if self._feedforward is not None:
            embedded_tokens = self._feedforward(embedded_tokens)
        ner_logits = self._tag_layer(embedded_tokens)

        viterbi_paths: List[Tuple[List[int], float]] = self._crf.viterbi_tags(
            ner_logits, ner_subtokens_mask
        )

        output = dict(
            classification_logits=classification_logits, viterbi_paths=viterbi_paths
        )

        if labels is not None and tags is not None:
            # Classification loss
            output["loss"] = self._classification_loss(classification_logits, labels)

            # NER loss

            # sort the ner_tokens_mask with all Trues first
            # pytorch has no stable sort, so we need a little trick
            trick = (
                torch.arange(
                    0,
                    1,
                    1 / ner_subtokens_mask.numel(),
                    device=ner_subtokens_mask.device,
                )  # in this way we avoid equal elements when sorting => no stable sort necessary
                .flip(-1)  # we want to sort with descending=True in the end
                .view(ner_subtokens_mask.size())
            )
            tags_mask, mask_indices = (ner_subtokens_mask + trick).sort(descending=True)
            # retype back to bool and truncate to the size of the tags
            tags_mask = tags_mask.int().bool()[:, : tags.size(-1)]

            # sort the ner_logits according to the mask indices
            first_index = torch.arange(
                ner_logits.size(0), device=ner_logits.device
            ).unsqueeze(1)
            ner_logits_sorted = ner_logits[first_index, mask_indices, :]
            # truncate to the size of the tags
            ner_logits_sorted = ner_logits_sorted[:, : tags.size(-1)]

            output["loss"] += -1 * self._crf(ner_logits_sorted, tags, tags_mask)

            # metrics
            self.metrics["classification_accuracy"](classification_logits, labels)
            self.metrics["classification_micro"](classification_logits, labels)
            self.metrics["classification_macro"](classification_logits, labels)
            self.metrics["ner_f1"](ner_logits_sorted, tags, tags_mask)

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

        pass

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
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
        for key, value in self.metrics["ner_f1"].get_metric(reset).items():
            metrics.update({f"ner/{key}": value})

        return metrics


class ProfNerTConfiguration(ComponentConfiguration[ProfNerT]):
    """Configuration for classification head components"""

    pass
