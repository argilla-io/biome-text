from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

import numpy
import torch
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.data.fields import TextField
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.util import get_text_field_mask
from captum.attr import IntegratedGradients

from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration
from biome.text.modules.configuration import Seq2VecEncoderConfiguration
from biome.text.modules.heads.classification.classification import ClassificationHead
from biome.text.modules.heads.task_prediction import Attribution
from biome.text.modules.heads.task_prediction import TextClassificationPrediction


class TextClassification(ClassificationHead):
    """
    Task head for text classification
    """

    label_name = "label"
    forward_arg_name = "text"

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        pooler: Optional[Seq2VecEncoderConfiguration] = None,
        feedforward: Optional[FeedForwardConfiguration] = None,
        multilabel: bool = False,
    ) -> None:

        super(TextClassification, self).__init__(backbone, labels, multilabel)

        self.pooler = (
            pooler.input_dim(self.backbone.encoder.get_output_dim()).compile()
            if pooler
            else BagOfEmbeddingsEncoder(self.backbone.encoder.get_output_dim())
        )
        self.feedforward = (
            None
            if not feedforward
            else feedforward.input_dim(self.pooler.get_output_dim()).compile()
        )
        self._classification_layer = torch.nn.Linear(
            (self.feedforward or self.pooler).get_output_dim(), self.num_labels
        )

    def featurize(
        self, text: Any, label: Optional[Union[int, str, List[Union[int, str]]]] = None
    ) -> Optional[Instance]:
        instance = self.backbone.featurizer(
            text,
            to_field=self.forward_arg_name,
            aggregate=True,
            exclude_record_keys=True,
        )
        return self._add_label(instance, label, to_field=self.label_name)

    def forward(  # type: ignore
        self,
        text: TextFieldTensors,
        label: torch.IntTensor = None,
    ) -> Dict[str, Any]:

        mask = get_text_field_mask(text)
        embeddings = self.backbone.embedder(text)
        logits = self._encoder_and_head_forward(embeddings, mask)

        output = self._make_forward_output(logits, label)

        # For computing the attributions
        # TODO: An optimized implementation would be to calculate the attributions directly in the forward method
        #  and provide a practical switch, maybe: `with head.turn_attributions_on(): self.forward_on_instances()`
        #  In this way we would calculate the attributions batch wise and on on GPU if available.
        output["embeddings"], output["mask"] = embeddings, mask

        return output

    def _encoder_and_head_forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """We reuse this method for the computation of the attributions"""
        encoded_text = self.pooler(
            self.backbone.encoder(embeddings, mask=mask), mask=mask
        )
        if self.feedforward:
            encoded_text = self.feedforward(encoded_text)

        return self._classification_layer(encoded_text)

    def _compute_attributions(
        self,
        single_forward_output,
        instance,
        n_steps: int = 50,
    ) -> List[Attribution]:
        """Attributes the prediction to the input.

        The attributions are calculated by means of the [Integrated Gradients](https://arxiv.org/abs/1703.01365) method.

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
            A list of attributions
        """
        # captum needs `torch.Tensor`s and we need a batch dimension (-> unsqueeze)
        embeddings = torch.from_numpy(single_forward_output["embeddings"]).unsqueeze(0)
        logits = torch.from_numpy(single_forward_output["logits"]).unsqueeze(0)
        mask = torch.from_numpy(single_forward_output["mask"]).unsqueeze(0)

        ig = IntegratedGradients(self._encoder_and_head_forward)
        attributions, delta = ig.attribute(
            embeddings,
            n_steps=n_steps,
            target=torch.argmax(logits),
            additional_forward_args=mask,
            return_convergence_delta=True,
        )
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().numpy()

        text_tokens = cast(TextField, instance[self.forward_arg_name]).tokens

        return [
            Attribution(
                text=token.text,
                start=token.idx,
                end=token.idx + len(token.text) if isinstance(token.idx, int) else None,
                field=self.forward_arg_name,
                attribution=attribution,
            )
            for token, attribution in zip(text_tokens, attributions)
        ]

    def _make_task_prediction(
        self,
        single_forward_output: Dict[str, numpy.ndarray],
        instance: Instance,
    ) -> TextClassificationPrediction:
        labels, probabilities = self._compute_labels_and_probabilities(
            single_forward_output
        )

        return TextClassificationPrediction(labels=labels, probabilities=probabilities)


class TextClassificationConfiguration(ComponentConfiguration[TextClassification]):
    """Configuration for classification head components"""

    pass
