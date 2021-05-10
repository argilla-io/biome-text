from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import numpy
import torch
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.nn import InitializerApplicator
from allennlp.nn import util
from captum.attr import IntegratedGradients

from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import BiMpmMatchingConfiguration
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2SeqEncoderConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.encoders import TimeDistributedEncoder
from biome.text.modules.heads.classification.classification import ClassificationHead
from biome.text.modules.heads.task_prediction import Attribution
from biome.text.modules.heads.task_prediction import RecordPairClassificationPrediction


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
    backbone
        Takes care of the embedding and optionally of the language encoding
    labels
        A list of labels for your classification task.
    field_encoder
        Encodes a data field, contextualized within the field
    record_encoder
        Encodes data fields, contextualized within the record
    matcher_forward
        BiMPM matching for the forward output of the record encoder layer
    matcher_backward
        BiMPM matching for the backward output of the record encoder layer
    aggregator
        Aggregator of all BiMPM matching vectors
    classifier_feedforward
        Fully connected layers for classification.
        A linear output layer with the number of labels at the end will be added automatically!!!
    dropout
        Dropout percentage to use.
    initializer
        If provided, will be used to initialize the model parameters.
    label_weights
        A list of weights for each label. The weights must be in the same order as the `labels`.
        You can also provide a dictionary that maps the label to its weight. Default: None.
    """

    _RECORD1_ARG_NAME_IN_FORWARD = "record1"
    _RECORD2_ARG_NAME_IN_FORWARD = "record2"

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
        label_weights: Optional[Union[List[float], Dict[str, float]]] = None,
    ):
        super().__init__(backbone, labels, label_weights=label_weights)

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
        record1: Dict[str, str],
        record2: Dict[str, str],
        label: Optional[str] = None,
    ) -> Instance:
        """Tokenizes, indexes and embeds the two records and optionally adds the label

        Parameters
        ----------
        record1
            First record
        record2
            Second record
        label
            Classification label

        Returns
        -------
        instance
            AllenNLP instance containing the two records plus optionally a label

        Raises
        ------
        FeaturizeError
        """
        record1_instance = self.backbone.featurizer(
            record1, to_field="record", aggregate=False
        )
        record2_instance = self.backbone.featurizer(
            record2, to_field="record", aggregate=False
        )

        instance = Instance(
            {
                self._RECORD1_ARG_NAME_IN_FORWARD: record1_instance.get("record"),
                self._RECORD2_ARG_NAME_IN_FORWARD: record2_instance.get("record"),
            }
        )

        return self._add_label(instance, label)

    def forward(
        self,  # type: ignore
        record1: TextFieldTensors,
        record2: TextFieldTensors,
        label: torch.LongTensor = None,
    ) -> Dict[str, Any]:
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

        output = self._make_forward_output(logits, label)

        # For computing the attributions
        # TODO: An optimized implementation would be to calculate the attributions directly in the forward method
        #  and provide a practical switch, maybe: `with head.turn_attributions_on(): self.forward_on_instances()`
        #  In this way we would calculate the attributions batch wise and on on GPU if available.
        if not self.training:
            output["field_encoded_record1"] = field_encoded_record1
            output["record_mask_record1"] = record_mask_record1
            output["field_encoded_record2"] = field_encoded_record2
            output["record_mask_record2"] = record_mask_record2

        return output

    def _field_encoding(
        self,
        record: TextFieldTensors,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embeds and encodes the records in a field context.

        We do this in a helper function to reuse it for the `self._compute_attributions` method.

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
        field_encoded_record: torch.Tensor = self._dropout(
            self._field_encoder(embedded_record, mask=field_mask)
        )

        # mask for the record encoder
        record_mask = cast(torch.BoolTensor, torch.sum(field_mask, -1) > 0)

        return field_encoded_record, record_mask

    def _bimpm_forward(
        self,
        field_encoded_record1: torch.Tensor,
        field_encoded_record2: torch.Tensor,
        record_mask_record1: torch.Tensor,
        record_mask_record2: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes in a record context, matches encodings, aggregates the matching results, classifies.

        We do this in a helper function to reuse it for the `self._compute_attributions` method.

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
            mask for encoded record1
        mask_record2
            mask for encoded record2

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

    def _compute_attributions(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
        n_steps: int = 50,
    ) -> List[Attribution]:
        """Computes attributions for each data field in the record by means of the
        [Integrated Gradients](https://arxiv.org/abs/1703.01365) method.

        IMPORTANT: The calculated attributions only make sense for a duplicate/not_duplicate binary classification task
        of the two records.

        Parameters
        ----------
        single_forward_output
            Non-batched forward output containing numpy arrays
        instance
            The instance containing the input data
        n_steps
            The number of steps used when calculating the attribution of each token.

        Returns
        -------
        attributions
        """
        # captum needs `torch.Tensor`s and we need a batch dimension (-> unsqueeze)
        field_encoded_record1 = torch.from_numpy(
            single_forward_output["field_encoded_record1"]
        ).unsqueeze(0)
        record_mask_record1 = torch.from_numpy(
            single_forward_output["record_mask_record1"]
        ).unsqueeze(0)

        field_encoded_record2 = torch.from_numpy(
            single_forward_output["field_encoded_record2"]
        ).unsqueeze(0)
        record_mask_record2 = torch.from_numpy(
            single_forward_output["record_mask_record2"]
        ).unsqueeze(0)

        logits = torch.from_numpy(single_forward_output["logits"]).unsqueeze(0)

        if not field_encoded_record1.size() == field_encoded_record2.size():
            raise RuntimeError("Both records must have the same number of data fields!")

        # 2. Get attributes
        ig = IntegratedGradients(self._bimpm_forward)

        prediction_target = torch.argmax(logits)

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
        #     raise RuntimeError("The `compute_attributions` method is only implemented for a binary classification task: "
        #                        "[duplicate, not_duplicate]")

        attributions_record1, delta_record1 = self._get_attributions_and_delta(
            ig_attribute_record1, 0
        )
        attributions_record2, delta_record2 = self._get_attributions_and_delta(
            ig_attribute_record2, 1
        )

        # 3. Get tokens corresponding to the attributions
        field_text_record1 = []
        for textfield in instance.get(self._RECORD1_ARG_NAME_IN_FORWARD):
            field_text_record1.append(
                " ".join([token.text for token in textfield.tokens])
            )
        field_text_record2 = []
        for textfield in instance.get(self._RECORD2_ARG_NAME_IN_FORWARD):
            field_text_record2.append(
                " ".join([token.text for token in textfield.tokens])
            )

        output_record1 = [
            Attribution(
                text=field_text,
                start=0,
                end=len(field_text),
                field=self._RECORD1_ARG_NAME_IN_FORWARD,
                attribution=attribution,
            )
            for field_text, attribution in zip(field_text_record1, attributions_record1)
        ]
        output_record2 = [
            Attribution(
                text=field_text,
                start=0,
                end=len(field_text),
                field=self._RECORD2_ARG_NAME_IN_FORWARD,
                attribution=attribution,
            )
            for field_text, attribution in zip(field_text_record2, attributions_record2)
        ]

        return output_record1 + output_record2

    @staticmethod
    def _get_attributions_and_delta(
        ig_attribute_output, zero_or_one: int
    ) -> Tuple[numpy.array, float]:
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

    def _make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
    ) -> RecordPairClassificationPrediction:
        labels, probabilities = self._compute_labels_and_probabilities(
            single_forward_output
        )

        return RecordPairClassificationPrediction(
            labels=labels, probabilities=probabilities
        )


class RecordPairClassificationConfiguration(
    ComponentConfiguration[RecordPairClassification]
):
    """Config for record pair classification head component"""

    pass
