from typing import Any, Dict, List, Optional

import torch
from allennlp.data import Instance
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
)
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.nn import InitializerApplicator, util
from overrides import overrides

from biome.text.backbone import ModelBackbone
from biome.text.modules.encoders import TimeDistributedEncoder
from biome.text.modules.heads.classification.defs import ClassificationHead
from biome.text.modules.specs import (
    BiMpmMatchingSpec,
    FeedForwardSpec,
    Seq2SeqEncoderSpec,
    Seq2VecEncoderSpec,
)


class BiMpm(ClassificationHead):
    """
    This ``Model`` is a version of AllenNLPs implementation of the BiMPM model described in
    `Bilateral Multi-Perspective Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_
    by Zhiguo Wang et al., 2017.

    This version adds the feature of being compatible with multiple inputs for the two records.
    The matching will be done for all possible combinations between the two records, that is:
    (r1_1, r2_1), (r1_1, r2_2), ..., (r1_2, r2_1), (r1_2, r2_2), ...

    This version also allows you to apply only one encoder, and to leave out the backward matching,
    providing the possibility to use transformers for the encoding layer.

    Parameters
    ----------
    backbone : `ModelBackbone`
        Takes care of the embedding
    labels : `List[str]`
        List of labels
    matcher_word : ``BiMpmMatchingSpec``
        BiMPM matching on the output of word embeddings of record1 and record2.
    encoder : ``Seq2SeqEncoderSpec``
        Encoder layer for record1 and record2
    matcher_forward : ``BiMPMMatching``
        BiMPM matching for the forward output of the encoder layer
    aggregator : ``Seq2VecEncoderSpec``
        Aggregator of all BiMPM matching vectors
    classifier_feedforward : ``FeedForwardSpec``
        Fully connected layers for classification.
        A linear output layer with the number of labels at the end will be added automatically!!!
    matcher_backward : ``BiMPMMatchingSpec``, optional
        BiMPM matching for the backward output of the encoder layer
    encoder2 : ``Seq2SeqEncoderSpec``, optional
        Encoder layer for encoded record1 and encoded record2
    matcher2_forward : ``BiMPMMatchingSpec``, optional
        BiMPM matching for the forward output of the second encoder layer
    matcher2_backward : ``BiMPMMatchingSpec``, optional
        BiMPM matching for the backward output of the second encoder layer
    dropout : ``float``, optional (default=0.1)
        Dropout percentage to use.
    multifield : ``bool``, optional (default=False)
        Are there multiple inputs for each record, that is do the inputs come from ``ListField``s?
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        matcher_word: BiMpmMatchingSpec,
        encoder: Seq2SeqEncoderSpec,
        matcher_forward: BiMpmMatchingSpec,
        aggregator: Seq2VecEncoderSpec,
        classifier_feedforward: FeedForwardSpec,
        matcher_backward: BiMpmMatchingSpec = None,
        encoder2: Seq2SeqEncoderSpec = None,
        matcher2_forward: BiMpmMatchingSpec = None,
        matcher2_backward: BiMpmMatchingSpec = None,
        dropout: float = 0.1,
        multifield: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
    ):
        super(BiMpm, self).__init__(backbone, labels)

        self.multifield = multifield
        self.num_wrapping_dims = 1 if multifield else 0
        self.matching = (
            self._multifield_matching if multifield else self._textfield_matching
        )
        if self.multifield:
            self.backbone.encoder = TimeDistributedEncoder(self.backbone.encoder)

        self.encoder: Seq2SeqEncoder = encoder.input_dim(
            self.backbone.encoder.get_output_dim()
        ).compile()
        if multifield:
            self.encoder: Seq2SeqEncoder = TimeDistributedEncoder(self.encoder)

        self.matcher_word: BiMpmMatching = matcher_word.input_dim(
            self.backbone.encoder.get_output_dim()
        ).compile()

        input_dim_matcher = self.encoder.get_output_dim()
        if matcher_backward is not None:
            input_dim_matcher //= 2

        self.matcher_forward: BiMpmMatching = matcher_forward.input_dim(
            input_dim_matcher
        ).compile()
        self.matcher_backward = (
            matcher_backward.input_dim(input_dim_matcher).compile()
            if matcher_backward is not None
            else None
        )

        self.encoder2 = (
            encoder2.input_dim(self.encoder.get_output_dim()).compile()
            if encoder2 is not None
            else None
        )
        if multifield and self.encoder2 is not None:
            self.encoder2 = TimeDistributedEncoder(self.encoder2)

        if self.encoder2 is not None:
            input_dim_matcher = self.encoder2.get_output_dim()
            if matcher2_backward is not None:
                input_dim_matcher //= 2

            self.matcher2_forward: BiMpmMatching = matcher2_forward.input_dim(
                input_dim_matcher
            ).compile()

            self.matcher2_backward = (
                matcher_backward.input_dim(input_dim_matcher).compile()
                if matcher_backward is not None
                else None
            )
        else:
            self.matcher2_forward = None
            self.matcher2_backward = None

        matching_dim = (
            self.matcher_word.get_output_dim() + self.matcher_forward.get_output_dim()
        )
        if self.matcher_backward:
            matching_dim += self.matcher_backward.get_output_dim()
        if self.matcher2_forward:
            matching_dim += self.matcher2_forward.get_output_dim()
        if self.matcher2_backward:
            matching_dim += self.matcher2_backward.get_output_dim()

        self.aggregator: Seq2VecEncoder = aggregator.input_dim(matching_dim).compile()

        self.classifier_feedforward: FeedForward = classifier_feedforward.input_dim(
            self.aggregator.get_output_dim() * 2
        ).compile()

        self.classifier_output_dim = self.classifier_feedforward.get_output_dim()
        self.output_layer = torch.nn.Linear(self.classifier_output_dim, self.num_labels)

        self.dropout = torch.nn.Dropout(dropout)

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def featurize(
        self,
        record1: Dict[str, Any],
        record2: Dict[str, Any],
        label: Optional[str] = None,
    ) -> Optional[Instance]:
        """Tokenizes, indexes and embedds the two records and optionally adds the label

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
        record1_field = self.backbone.featurize(record1, aggregate=not self.multifield)
        record2_field = self.backbone.featurize(record2, aggregate=not self.multifield)
        instance = Instance(
            {
                "record1": record1_field.get("record"),
                "record2": record2_field.get("record"),
            }
        )
        return self.add_label(instance, label)

    @overrides
    def forward(
        self,  # type: ignore
        record1: Dict[str, torch.LongTensor],
        record2: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        record1
            The first input tokens.
            The dictionary is the output of a ``*Field.as_array()``. It gives names to the tensors created by
            the ``TokenIndexer``s.
            In its most basic form, using a ``SingleIdTokenIndexer``, the dictionary is composed of:
            ``{"tokens": Tensor(batch_size, num_tokens)}``.
            The keys of the dictionary are defined in the `pipeline.yaml` config.
            The dictionary is designed to be passed on directly to a ``TextFieldEmbedder``, that has a
            ``TokenEmbedder`` for each key in the dictionary (except you set `allow_unmatched_keys` in the
            ``TextFieldEmbedder`` to False) and knows how to combine different word/character representations into a
            single vector per token in your input.
        record2
            The second input tokens.
        label : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class label of shape
            ``(batch_size, num_classes)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        mask_record1 = util.get_text_field_mask(
            record1, num_wrapping_dims=self.num_wrapping_dims
        )
        mask_record2 = util.get_text_field_mask(
            record2, num_wrapping_dims=self.num_wrapping_dims
        )

        # embedding and encoding of record1
        embedded_record1 = self.dropout(
            self.backbone.forward(
                record1, mask=mask_record1, num_wrapping_dims=self.num_wrapping_dims
            )
        )
        encoded_record1 = self.dropout(
            self.encoder(embedded_record1, mask=mask_record1)
        )
        encoded2_record1 = (
            self.dropout(self.encoder2(encoded_record1, mask=mask_record1))
            if self.encoder2
            else None
        )

        # embedding and encoding of record2
        embedded_record2 = self.dropout(
            self.backbone.forward(
                record2, mask=mask_record2, num_wrapping_dims=self.num_wrapping_dims
            )
        )
        encoded_record2 = self.dropout(
            self.encoder(embedded_record2, mask=mask_record2)
        )
        encoded2_record2 = (
            self.dropout(self.encoder2(encoded_record2, mask=mask_record2))
            if self.encoder2
            else None
        )

        aggregated_records_cat = self.matching(
            embedded_record1,
            embedded_record2,
            mask_record1,
            mask_record2,
            encoded_record1,
            encoded_record2,
            encoded2_record1,
            encoded2_record2,
        )

        # the final forward layer
        logits = self.output_layer(self.classifier_feedforward(aggregated_records_cat))

        return self.single_label_output(logits, label)

    def _textfield_matching(
        self,
        embedded_record1,
        embedded_record2,
        mask_record1,
        mask_record2,
        encoded_record1,
        encoded_record2,
        encoded2_record1,
        encoded2_record2,
    ):
        """Implements the Matching layer in case of a TextField input"""
        matching_vector_record1: List[torch.Tensor] = []
        matching_vector_record2: List[torch.Tensor] = []

        def add_matching_result(matcher, record1_values, record2_values):
            # utility function to get matching result and add to the result list
            matching_result = matcher(
                record1_values, mask_record1, record2_values, mask_record2
            )
            matching_vector_record1.extend(matching_result[0])
            matching_vector_record2.extend(matching_result[1])

        # calculate matching vectors from word embedding, first layer encoding, and second layer encoding
        add_matching_result(self.matcher_word, embedded_record1, embedded_record2)

        half_hidden_size = None
        if self.matcher_backward is not None:
            half_hidden_size = self.encoder.get_output_dim() // 2

        half_hidden_size2 = None
        if self.matcher2_backward is not None:
            half_hidden_size2 = self.encoder2.get_output_dim() // 2

        add_matching_result(
            self.matcher_forward,
            encoded_record1[:, :, :half_hidden_size],
            encoded_record2[:, :, :half_hidden_size],
        )
        if self.matcher_backward is not None:
            add_matching_result(
                self.matcher_backward,
                encoded_record1[:, :, half_hidden_size:],
                encoded_record2[:, :, half_hidden_size:],
            )

        if self.encoder2 is not None:
            add_matching_result(
                self.matcher2_forward,
                encoded2_record1[:, :, :half_hidden_size2],
                encoded2_record2[:, :, :half_hidden_size2],
            )

            if self.matcher2_backward is not None:
                add_matching_result(
                    self.matcher2_backward,
                    encoded2_record1[:, :, half_hidden_size2:],
                    encoded2_record2[:, :, half_hidden_size2:],
                )

        # concat the matching vectors
        matching_vector_cat_record1 = self.dropout(
            torch.cat(matching_vector_record1, dim=2)
        )
        matching_vector_cat_record2 = self.dropout(
            torch.cat(matching_vector_record2, dim=2)
        )

        # aggregate the matching vectors
        aggregated_record1 = self.dropout(
            self.aggregator(matching_vector_cat_record1, mask_record1)
        )
        aggregated_record2 = self.dropout(
            self.aggregator(matching_vector_cat_record2, mask_record2)
        )

        return torch.cat([aggregated_record1, aggregated_record2], dim=-1)

    def _multifield_matching(
        self,
        embedded_record1,
        embedded_record2,
        mask_record1,
        mask_record2,
        encoded_record1,
        encoded_record2,
        encoded2_record1,
        encoded2_record2,
    ):
        """Implements the Matching layer in case of a ListField input"""
        multifield_matching_vector_record1: List[torch.Tensor] = []
        multifield_matching_vector_record2: List[torch.Tensor] = []
        multifield_matching_mask_record1: List[torch.Tensor] = []
        multifield_matching_mask_record2: List[torch.Tensor] = []

        def add_matching_result(
            matcher,
            record1_values,
            record1_mask,
            record2_values,
            record2_mask,
            record1_list,
            record2_list,
        ):
            # utility function to get matching result and add to the result list
            matching_result = matcher(
                record1_values, record1_mask, record2_values, record2_mask
            )
            record1_list.extend(matching_result[0])
            record2_list.extend(matching_result[1])

        # calculate matching vectors from word embedding and layer encoding
        half_hidden_size = None
        if self.matcher_backward is not None:
            half_hidden_size = self.encoder.get_output_dim() // 2
        half_hidden_size2 = None
        if self.matcher2_backward is not None:
            half_hidden_size2 = self.encoder2.get_output_dim() // 2

        for i in range(embedded_record1.shape[1]):
            for j in range(embedded_record2.shape[1]):
                matching_vector_record1: List[torch.Tensor] = []
                matching_vector_record2: List[torch.Tensor] = []

                add_matching_result(
                    self.matcher_word,
                    embedded_record1[:, i, :, :],
                    mask_record1[:, i, :],
                    embedded_record2[:, j, :, :],
                    mask_record2[:, j, :],
                    matching_vector_record1,
                    matching_vector_record2,
                )
                add_matching_result(
                    self.matcher_forward,
                    encoded_record1[:, i, :, :half_hidden_size],
                    mask_record1[:, i, :],
                    encoded_record2[:, j, :, :half_hidden_size],
                    mask_record2[:, j, :],
                    matching_vector_record1,
                    matching_vector_record2,
                )
                if self.matcher_backward:
                    add_matching_result(
                        self.matcher_backward,
                        encoded_record1[:, i, :, half_hidden_size:],
                        mask_record1[:, i, :],
                        encoded_record2[:, j, :, half_hidden_size:],
                        mask_record2[:, j, :],
                        matching_vector_record1,
                        matching_vector_record2,
                    )

                if self.encoder2:
                    add_matching_result(
                        self.matcher2_forward,
                        encoded2_record1[:, i, :, :half_hidden_size2],
                        mask_record1[:, i, :],
                        encoded2_record2[:, j, :, :half_hidden_size2],
                        mask_record2[:, j, :],
                        matching_vector_record1,
                        matching_vector_record2,
                    )
                    if self.matcher2_backward:
                        add_matching_result(
                            self.matcher2_backward,
                            encoded2_record1[:, i, :, half_hidden_size2:],
                            mask_record1[:, i, :],
                            encoded2_record2[:, j, :, half_hidden_size2:],
                            mask_record2[:, j, :],
                            matching_vector_record1,
                            matching_vector_record2,
                        )

                # concat the matching vectors
                matching_vector_cat_record1 = self.dropout(
                    torch.cat(matching_vector_record1, dim=2)
                )
                matching_vector_cat_record2 = self.dropout(
                    torch.cat(matching_vector_record2, dim=2)
                )

                multifield_matching_vector_record1.append(matching_vector_cat_record1)
                multifield_matching_vector_record2.append(matching_vector_cat_record2)
                multifield_matching_mask_record1.append(mask_record1[:, i, :])
                multifield_matching_mask_record2.append(mask_record2[:, j, :])

        # concat the multifield vectors and masks
        multifield_matching_vector_cat_record1 = torch.cat(
            multifield_matching_vector_record1, dim=1
        )
        multifield_matching_vector_cat_record2 = torch.cat(
            multifield_matching_vector_record2, dim=1
        )
        multifield_matching_mask_cat_record1 = torch.cat(
            multifield_matching_mask_record1, dim=1
        )
        multifield_matching_mask_cat_record2 = torch.cat(
            multifield_matching_mask_record2, dim=1
        )

        # aggregate the matching vectors
        aggregated_record1 = self.dropout(
            self.aggregator(
                multifield_matching_vector_cat_record1,
                multifield_matching_mask_cat_record1,
            )
        )
        aggregated_record2 = self.dropout(
            self.aggregator(
                multifield_matching_vector_cat_record2,
                multifield_matching_mask_cat_record2,
            )
        )

        return torch.cat([aggregated_record1, aggregated_record2], dim=-1)