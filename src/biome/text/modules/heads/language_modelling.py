from typing import Dict, Optional

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Perplexity

from biome.text.featurizer import InputFeaturizer
from biome.text.backbone import BackboneEncoder
from biome.text.modules.specs import ComponentSpec
from biome.text.vocabulary import vocabulary
from .defs import TaskHead, TaskName, TaskOutput


class SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    TODO: copied from allennlp master branch, remove when 1.0 is released
    """

    def __init__(self, num_words: int, embedding_dim: int) -> None:
        super(SoftmaxLoss, self).__init__()

        self.softmax_w = torch.nn.Parameter(
            torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(
            torch.matmul(embeddings, self.softmax_w) + self.softmax_b, dim=-1
        )

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")


class LanguageModelling(TaskHead):
    """
    Task head for next-token language modelling, i.e., a model to predict the next token
    in a sequence of tokens.
    """

    def task_name(self) -> TaskName:
        return TaskName.language_modelling

    def __init__(self, backbone: BackboneEncoder, dropout: float = None) -> None:
        super(LanguageModelling, self).__init__(backbone)

        if not backbone.featurizer.words:
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

    def _on_vocab_update(self):
        self._loss = SoftmaxLoss(
            num_words=vocabulary.words_vocab_size(self.backbone.vocab),
            embedding_dim=self.backbone.encoder.get_output_dim(),
        )

    def featurize(self, text: str) -> Optional[Instance]:
        return self.backbone.featurize(text, to_field="text", aggregate=True)

    def forward(  # type: ignore
        self, text: Dict[str, torch.Tensor]
    ) -> TaskOutput:

        mask = get_text_field_mask(text)
        contextual_embeddings = self.backbone.forward(text, mask)

        token_ids = text.get(InputFeaturizer.WORDS)
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


class LanguageModellingSpec(ComponentSpec[LanguageModelling]):
    """Spec for language model head components"""

    pass
