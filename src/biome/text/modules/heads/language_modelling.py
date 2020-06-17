from typing import Dict, Optional

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance, TextFieldTensors
from allennlp.modules import SoftmaxLoss
from allennlp.nn.util import get_text_field_mask, get_token_ids_from_text_field_tensors
from allennlp.training.metrics import Perplexity

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration
from .task_head import TaskHead, TaskName, TaskOutput


class LanguageModelling(TaskHead):
    """
    Task head for next-token language modelling, i.e., a model to predict the next token
    in a sequence of tokens.
    """

    def task_name(self) -> TaskName:
        return TaskName.language_modelling

    def __init__(self, backbone: ModelBackbone, dropout: float = None) -> None:
        super(LanguageModelling, self).__init__(backbone)

        if not backbone.featurizer.has_word_features:
            raise ConfigurationError(
                "`LanguageModelling` defines a word-level next token language model. "
                "Please check your `features` configuration to enable at least `words` features."
            )

        self._forward_dim = backbone.encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self.metrics = {"perplexity": Perplexity()}

        self._loss = SoftmaxLoss(
            num_words=vocabulary.words_vocab_size(self.backbone.vocab),
            embedding_dim=self.backbone.encoder.get_output_dim(),
        )

    def on_vocab_update(self):
        self._loss = SoftmaxLoss(
            num_words=vocabulary.words_vocab_size(self.backbone.vocab),
            embedding_dim=self.backbone.encoder.get_output_dim(),
        )

    def featurize(self, text: str) -> Optional[Instance]:
        return self.backbone.featurizer(text, to_field="text", aggregate=True)

    def forward(  # type: ignore
        self, text: TextFieldTensors
    ) -> TaskOutput:

        mask = get_text_field_mask(text)
        contextual_embeddings = self.backbone.forward(text, mask)
        # NOTE: @dvsrepo, Allennlp 1.0 includes a second features level that I'm not sure of understand.
        # Anyway, they proved a function to realize the target here (the function docstring clarifies the
        # real spaghetti inside indexer code references, :-)
        token_ids = get_token_ids_from_text_field_tensors(text)
        assert isinstance(contextual_embeddings, torch.Tensor)

        # Use token_ids to compute targets
        # targets are next token ids with respect to first token in the seq
        # e.g. token_ids [[1, 3, 5, 7],..[]], forward_targets=[[3,5,7],..]
        forward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]

        # add dropout
        contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

        # compute softmax loss
        try:
            forward_loss = self._compute_loss(
                contextual_embeddings_with_dropout, forward_targets
            )
        except IndexError:
            raise IndexError(
                "Word token out of vocabulary boundaries, please check your vocab is correctly set"
                " or created before starting training."
            )

        num_targets = torch.sum((forward_targets > 0).long())
        if num_targets > 0:
            average_loss = forward_loss / num_targets.float()
        else:
            average_loss = torch.tensor(0.0).to(forward_targets.device)

        for metric in self.metrics.values():
            metric(average_loss)

        return TaskOutput(
            logits=None,
            probs=None,
            loss=average_loss,
            **{"lm_embeddings": contextual_embeddings, "mask": mask}
        )

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

    def _compute_loss(
        self, lm_embeddings: torch.Tensor, forward_targets: torch.Tensor
    ) -> torch.Tensor:
        forward_embeddings = lm_embeddings
        forward_loss = self._loss_helper(0, forward_embeddings, forward_targets)
        return forward_loss

    def _loss_helper(
        self,
        direction: int,
        direction_embeddings: torch.Tensor,
        direction_targets: torch.Tensor,
    ) -> torch.Tensor:
        mask = direction_targets > 0
        # we need to subtract 1 to undo the padding id since the softmax
        # does not include a padding dimension

        # shape (batch_size * timesteps, )
        non_masked_targets = direction_targets.masked_select(mask) - 1

        # shape (batch_size * timesteps, embedding_dim)
        non_masked_embeddings = direction_embeddings.masked_select(
            mask.unsqueeze(-1)
        ).view(-1, self._forward_dim)
        return self._loss(non_masked_embeddings, non_masked_targets)


class LanguageModellingConfiguration(ComponentConfiguration[LanguageModelling]):
    """Spec for language model head components"""

    pass
