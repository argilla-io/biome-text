from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from allennlp.data import Batch, Instance, TextFieldTensors
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
)
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.nn import InitializerApplicator, util
from captum.attr import IntegratedGradients

from biome.text.backbone import ModelBackbone
from biome.text.configuration import CharFeatures, WordFeatures
from biome.text.helpers import (
    get_char_tokens_ids_from_text_field_tensors,
    get_word_tokens_ids_from_text_field_tensors,
)
from biome.text.modules.encoders import TimeDistributedEncoder
from biome.text.modules.configuration import (
    BiMpmMatchingConfiguration,
    FeedForwardConfiguration,
    Seq2SeqEncoderConfiguration,
    Seq2VecEncoderConfiguration,
    ComponentConfiguration,
)
from .classification import ClassificationHead
from ..task_head import TaskOutput


class RecordPairClassification(ClassificationHead):
    """Classifies the relation between a pair of records using a matching layer.

    The input for models using this `TaskHead` are two *records* with one or more *data fields* each, and a label
    describing their relationship.
    If you would like a meaningful explanation of the model's prediction,
    both records must consist of the same number of *data fields* and hold them in the same order.

    The architecture is loosely based on the AllenNLP implementation of the BiMPM model described in
    `Bilateral Multi-Perspective Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_
    by Zhiguo Wang et al., 2017, and was adapted to deal with record pairs.

    Parameters
    ----------
    backbone : `ModelBackbone`
        Takes care of the embedding and optionally of the language encoding
    labels : `List[str]`
        List of labels
    field_encoder : `Seq2VecEncoder`
        Encodes a data field, contextualized within the field
    record_encoder : `Seq2SeqEncoder`
        Encodes data fields, contextualized within the record
    matcher_forward : `BiMPMMatching`
        BiMPM matching for the forward output of the record encoder layer
    matcher_backward : `BiMPMMatching`, optional
        BiMPM matching for the backward output of the record encoder layer
    aggregator : `Seq2VecEncoder`
        Aggregator of all BiMPM matching vectors
    classifier_feedforward : `FeedForward`
        Fully connected layers for classification.
        A linear output layer with the number of labels at the end will be added automatically!!!
    dropout : ``float``, optional (default=0.1)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        field_encoder: Seq2VecEncoderConfiguration,
        record_encoder: Seq2SeqEncoderConfiguration,
        matcher_forward: BiMpmMatchingConfiguration,
        aggregator: Seq2VecEncoderConfiguration,
        classifier_feedforward: FeedForwardConfiguration,
        matcher_backward: BiMpmMatchingConfiguration = None,
        dropout: float = 0.1,
        initializer: InitializerApplicator = InitializerApplicator(),
    ):
        super(RecordPairClassification, self).__init__(backbone, labels)

        # This is needed for the TrainerConfig to choose the right 'sorting_keys'
        self.backbone.encoder = TimeDistributedEncoder(self.backbone.encoder)

        self._field_encoder: Seq2VecEncoder = field_encoder.input_dim(
            self.backbone.encoder.get_output_dim()
        ).compile()
        self._field_encoder = TimeDistributedEncoder(self._field_encoder)

        self._record_encoder: Seq2SeqEncoder = record_encoder.input_dim(
            self._field_encoder.get_output_dim()
        ).compile()

        input_dim_matcher = self._record_encoder.get_output_dim()
        if matcher_backward is not None:
            input_dim_matcher //= 2

        self._matcher_forward: BiMpmMatching = matcher_forward.input_dim(
            input_dim_matcher
        ).compile()
        self._matcher_backward = (
            matcher_backward.input_dim(input_dim_matcher).compile()
            if matcher_backward is not None
            else None
        )

        matching_dim = self._matcher_forward.get_output_dim()
        if self._matcher_backward:
            matching_dim += self._matcher_backward.get_output_dim()

        self._aggregator: Seq2VecEncoder = aggregator.input_dim(matching_dim).compile()

        # we aggregate the two records in the end -> *2
        self._classifier_feedforward: FeedForward = classifier_feedforward.input_dim(
            self._aggregator.get_output_dim() * 2
        ).compile()

        self._output_layer = torch.nn.Linear(
            self._classifier_feedforward.get_output_dim(), self.num_labels
        )

        self._dropout = torch.nn.Dropout(dropout)

        initializer(self)

    def featurize(
        self,
        record1: Dict[str, Any],
        record2: Dict[str, Any],
        label: Optional[str] = None,
    ) -> Optional[Instance]:
        """Tokenizes, indexes and embeds the two records and optionally adds the label

        Parameters
        ----------
        record1 : Dict[str, Any]
            First record
        record2 : Dict[str, Any]
            Second record
        label : Optional[str]
            Classification label

        Returns
        -------
        instance
            AllenNLP instance containing the two records plus optionally a label
        """
        record1_instance = self.backbone.featurizer(
            record1, to_field="record", aggregate=False
        )
        record2_instance = self.backbone.featurizer(
            record2, to_field="record", aggregate=False
        )
        instance = Instance(
            {
                "record1": record1_instance.get("record"),
                "record2": record2_instance.get("record"),
            }
        )
        instance = self.add_label(instance, label)

        return instance

    def forward(
        self,  # type: ignore
        record1: TextFieldTensors,
        record2: TextFieldTensors,
        label: torch.LongTensor = None,
    ) -> TaskOutput:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        record1
            Tokens of the first record.
            The dictionary is the output of a `ListField.as_array()`. It gives names to the tensors created by
            the `TokenIndexer`s.
            In its most basic form, using a `SingleIdTokenIndexer`, the dictionary is composed of:
            `{"tokens": {"tokens": Tensor(batch_size, num_fields, num_tokens)}}`.
            The dictionary is designed to be passed on directly to a `TextFieldEmbedder`, that has a
            `TokenEmbedder` for each key in the dictionary (except you set `allow_unmatched_keys` in the
            `TextFieldEmbedder` to False) and knows how to combine different word/character representations into a
            single vector per token in your input.
        record2
            Tokens of the second record.
        label : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class label of shape
            `(batch_size, num_classes)`.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
        class_probabilities : torch.FloatTensor
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # embeding and encoding (field context)
        field_encoded_record1, record_mask_record1 = self._field_encoding(record1)
        field_encoded_record2, record_mask_record2 = self._field_encoding(record2)

        # encoding (record context), matching, aggregation, classification
        logits = self._bimpm_forward(
            field_encoded_record1,
            field_encoded_record2,
            record_mask_record1,
            record_mask_record2,
        )

        output: TaskOutput = self.single_label_output(logits, label)

        return output

    def _field_encoding(
        self, record: TextFieldTensors,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embeds and encodes the records in a field context.

        We do this in a helper function to reuse it for the `explain_prediction` method.

        Parameters
        ----------
        record
            See `self.forward`

        Returns
        -------
        field_encoded_record
        record_mask
        """
        field_mask = util.get_text_field_mask(record, num_wrapping_dims=1)

        # embedding and encoding of one record (field context)
        embedded_record = self._dropout(
            self.backbone.forward(record, mask=field_mask, num_wrapping_dims=1)
        )
        field_encoded_record = self._dropout(
            self._field_encoder(embedded_record, mask=field_mask)
        )

        # mask for the record encoder
        record_mask = torch.sum(field_mask, -1) > 0

        return field_encoded_record, record_mask

    def _bimpm_forward(
        self,
        field_encoded_record1: torch.Tensor,
        field_encoded_record2: torch.Tensor,
        record_mask_record1: torch.Tensor,
        record_mask_record2: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes in a record context, matches encodings, aggregates the matching results, classifies.

        We do this in a helper function to reuse it for the `explain_prediction` method.

        Parameters
        ----------
        field_encoded_record1
            Encoded record1 (in a field context)
        field_encoded_record2
            Encoded record2 (in a field context)
        record_mask_record1
            Mask for the record encoder for record1
        record_mask_record2
            Mask for the record encoder for record2

        Returns
        -------
        logits
        """
        record_encoded_record1 = self._dropout(
            self._record_encoder(field_encoded_record1, mask=record_mask_record1)
        )

        record_encoded_record2 = self._dropout(
            self._record_encoder(field_encoded_record2, mask=record_mask_record2)
        )

        # matching layer
        matching_vector_record1, matching_vector_record2 = self._matching_layer(
            record_encoded_record1,
            record_encoded_record2,
            record_mask_record1,
            record_mask_record2,
        )

        # aggregation layer
        aggregated_record1 = self._dropout(
            self._aggregator(matching_vector_record1, record_mask_record1)
        )
        aggregated_record2 = self._dropout(
            self._aggregator(matching_vector_record2, record_mask_record2)
        )

        aggregated_records = torch.cat([aggregated_record1, aggregated_record2], dim=-1)

        # the final feed forward layer
        logits = self._output_layer(self._classifier_feedforward(aggregated_records))

        return logits

    def _matching_layer(
        self,
        encoded_record1: torch.Tensor,
        encoded_record2: torch.Tensor,
        mask_record1: torch.Tensor,
        mask_record2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implements the Matching layer

        Parameters
        ----------
        encoded_record1
            encoded record1 (record context)
        encoded_record2
            encoded record2 (record context)
        mask_record1
            mask for record encoder for record1
        mask_record2
            mask for record encoder for record2

        Returns
        -------
        matching_vector_record1_cat
            Concatenated matching results for record1
        matching_vector_record2_cat
            Concatenated matching results for record2
        """
        matching_vector_record1: List[torch.Tensor] = []
        matching_vector_record2: List[torch.Tensor] = []

        # Matching is done separately for forward and backward encodings (if backward is specified)
        half_hidden_size = None
        if self._matcher_backward is not None:
            half_hidden_size = self._record_encoder.get_output_dim() // 2

        # forward matching
        matching_result = self._matcher_forward(
            encoded_record1[:, :, :half_hidden_size],
            mask_record1,
            encoded_record2[:, :, :half_hidden_size],
            mask_record2,
        )
        matching_vector_record1.extend(matching_result[0])
        matching_vector_record2.extend(matching_result[1])

        # backward matching
        if self._matcher_backward is not None:
            matching_result = self._matcher_backward(
                encoded_record1[:, :, half_hidden_size:],
                mask_record1,
                encoded_record2[:, :, half_hidden_size:],
                mask_record2,
            )
            matching_vector_record1.extend(matching_result[0])
            matching_vector_record2.extend(matching_result[1])

        # concat the matching vectors + apply dropout
        matching_vector_record1_cat = self._dropout(
            torch.cat(matching_vector_record1, dim=2)
        )
        matching_vector_record2_cat = self._dropout(
            torch.cat(matching_vector_record2, dim=2)
        )

        return matching_vector_record1_cat, matching_vector_record2_cat

    def explain_prediction(
        self, prediction: Dict[str, np.array], instance: Instance, n_steps: int
    ) -> Dict[str, Any]:
        """Calculates attributions for each data field in the record by integrating the gradients.

        IMPORTANT: The calculated attributions only make sense for a duplicate/not_duplicate binary classification task
        of the two records.

        Parameters
        ----------
        prediction
        instance
        n_steps

        Returns
        -------
        prediction_dict
            The prediction dictionary with a newly added "explain" key
        """
        # TODO(dcfidalgo): optimize: for the prediction we already embedded and field encoded the records.
        #     Also, the forward passes here are always done on cpu!

        batch = Batch([instance])
        tokens_ids = batch.as_tensor_dict()

        # get attributions for each field
        field_encoded_record1, record_mask_record1 = self._field_encoding(
            tokens_ids.get("record1")
        )
        field_encoded_record2, record_mask_record2 = self._field_encoding(
            tokens_ids.get("record2")
        )
        if not field_encoded_record2.size() == field_encoded_record2.size():
            raise RuntimeError("Both records must have the same number of data fields!")

        ig = IntegratedGradients(self._bimpm_forward)

        prediction_target = int(np.argmax(prediction["probs"]))
        ig_attribute_record1 = ig.attribute(
            inputs=(field_encoded_record1, field_encoded_record2),
            baselines=(field_encoded_record2, field_encoded_record2),
            additional_forward_args=(record_mask_record1, record_mask_record2),
            target=prediction_target,
            return_convergence_delta=True,
            n_steps=n_steps,
        )

        ig_attribute_record2 = ig.attribute(
            inputs=(field_encoded_record1, field_encoded_record2),
            baselines=(field_encoded_record1, field_encoded_record1),
            additional_forward_args=(record_mask_record1, record_mask_record2),
            target=prediction_target,
            return_convergence_delta=True,
            n_steps=n_steps,
        )
        # The code below was an attempt to make attributions for the "duplicate case" more meaningful ... did not work
        # # duplicate case:
        # # Here we integrate each record along the path from the null vector -> record1/2
        # # assuming that the null vector provides the highest "not duplicate" score.
        # if prediction_target == 0:
        #     ig_attribute_record1 = ig.attribute(
        #         inputs=(field_encoded_record1, field_encoded_record2),
        #         baselines=(torch.zeros_like(field_encoded_record1), field_encoded_record2),
        #         additional_forward_args=(record_mask_record1, record_mask_record2),
        #         # we fix the target since we want negative integrals always to be associated
        #         # to the "not_duplicate" case and positive ones to the "duplicate" case
        #         target=0,
        #         return_convergence_delta=True,
        #     )
        #
        #     ig_attribute_record2 = ig.attribute(
        #         inputs=(field_encoded_record1, field_encoded_record2),
        #         baselines=(field_encoded_record1, torch.zeros_like(field_encoded_record2)),
        #         additional_forward_args=(record_mask_record1, record_mask_record2),
        #         # we fix the target since we want negative integrals always to be associated
        #         # to the "not_duplicate" case and positive ones to the "duplicate" case
        #         target=0,
        #         return_convergence_delta=True,
        #     )
        #
        # # not duplicate case:
        # # Here we integrate each record along the path from record2/1 -> record1/2
        # # assuming that the same record provides the highest "duplicate" score.
        # elif prediction_target == 1:
        #     ...
        # else:
        #     raise RuntimeError("The `explain` method is only implemented for a binary classification task: "
        #                        "[duplicate, not_duplicate]")

        attributions_record1, delta_record1 = self._get_attributions_and_delta(
            ig_attribute_record1, 0
        )
        attributions_record2, delta_record2 = self._get_attributions_and_delta(
            ig_attribute_record2, 1
        )

        # get tokens corresponding to the attributions
        field_tokens_record1 = self._get_field_tokens(tokens_ids.get("record1"))
        field_tokens_record2 = self._get_field_tokens(tokens_ids.get("record2"))

        return {
            **prediction,
            "explain": {
                "record1": [
                    {"token": token, "attribution": attribution}
                    for token, attribution in zip(
                        field_tokens_record1, attributions_record1
                    )
                ],
                "record2": [
                    {"token": token, "attribution": attribution}
                    for token, attribution in zip(
                        field_tokens_record2, attributions_record2
                    )
                ],
            },
        }

    def _get_field_tokens(self, record_token_ids: TextFieldTensors) -> List[str]:
        """
        TODO(dcfidalgo): This can very likely be optimised!
        Parameters
        ----------
        record_token_ids

        Returns
        -------

        """
        field_tokens = []

        if WordFeatures.namespace in record_token_ids:
            # batch size is 1 -> [0]
            for field in get_word_tokens_ids_from_text_field_tensors(record_token_ids)[
                0
            ]:
                tokens = []
                for word_idx in field:
                    # skipp padding
                    if word_idx.item() == 0:
                        continue
                    token = self.backbone.vocab.get_token_from_index(
                        word_idx.item(), namespace=WordFeatures.namespace
                    )
                    tokens.append(token)
                field_tokens.append(" ".join(tokens))

        elif CharFeatures.namespace in record_token_ids:
            # batch size is 1 -> [0]
            for field in get_char_tokens_ids_from_text_field_tensors(record_token_ids)[
                0
            ]:
                tokens = []
                for word in field:
                    token = []
                    for char_idx in word:
                        # skipp padding
                        if char_idx.item() == 0:
                            continue
                        char = self.backbone.vocab.get_token_from_index(
                            char_idx.item(), namespace=CharFeatures.namespace
                        )
                        token.append(char)
                    if token:
                        tokens.append("".join(token))
                field_tokens.append(" ".join(tokens))

        return field_tokens

    @staticmethod
    def _get_attributions_and_delta(
        ig_attribute_output, zero_or_one: int
    ) -> Tuple[np.array, float]:
        """Gets attributions and delta out of the `IntegratedGradients.attribute()` output.

        Parameters
        ----------
        ig_attribute_output
            Output of the `IntegratedGradients.attribute()` method.
        zero_or_one
            Is either 0 for record1 or 1 for record2

        Returns
        -------
        attributions, delta
        """
        attributions = ig_attribute_output[0][zero_or_one].sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        delta = ig_attribute_output[1]

        return attributions, delta


class RecordPairClassificationConfiguration(
    ComponentConfiguration[RecordPairClassification]
):
    """Config for record pair classification head component"""

    pass
