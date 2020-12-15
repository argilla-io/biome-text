from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.modules import SoftmaxLoss
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_token_ids_from_text_field_tensors
from allennlp.training.metrics import Perplexity

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.modules.configuration import ComponentConfiguration

from .task_head import TaskHead
from .task_head import TaskName
from .task_head import TaskOutput


class LanguageModelling(TaskHead):
    """
    Task head for next-token language modelling, i.e., a model to predict the next token
    in a sequence of tokens.
    """

    task_name = TaskName.language_modelling

    def __init__(
        self,
        backbone: ModelBackbone,
        dropout: float = None,
        bidirectional: bool = False,
    ) -> None:
        super(LanguageModelling, self).__init__(backbone)

        self.bidirectional = bidirectional

        if not backbone.featurizer.has_word_features:
            raise ConfigurationError(
                "`LanguageModelling` defines a word-level next token language model. "
                "Please check your `features` configuration to enable at least `words` features."
            )

        if backbone.encoder.is_bidirectional() is not bidirectional:
            raise ConfigurationError(
                "Bidirectionality of contextualizer must match bidirectionality of "
                "language model. "
                f"Contextualizer bidirectional: {backbone.encoder.is_bidirectional()}, "
                f"language model bidirectional: {bidirectional}"
            )

        if self.bidirectional:
            self._forward_dim = backbone.encoder.get_output_dim() // 2
        else:
            self._forward_dim = backbone.encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self.metrics = {"perplexity": Perplexity()}

        self._loss = SoftmaxLoss(
            num_words=vocabulary.words_vocab_size(self.backbone.vocab),
            embedding_dim=self._forward_dim,
        )

    def on_vocab_update(self):
        num_words = vocabulary.words_vocab_size(self.backbone.vocab)
        if len(self._loss.softmax_b) != num_words:
            self._loss = SoftmaxLoss(
                num_words=num_words,
                embedding_dim=self._forward_dim,
            )

    def featurize(self, text: str) -> Optional[Instance]:
        return self.backbone.featurizer(text, to_field="text", aggregate=True)

    def forward(self, text: TextFieldTensors) -> TaskOutput:  # type: ignore

        mask = get_text_field_mask(text)
        contextual_embeddings = self.backbone.forward(text, mask)

        token_ids = get_token_ids_from_text_field_tensors(text)
        assert isinstance(contextual_embeddings, torch.Tensor)

        # Use token_ids to compute targets
        # targets are next token ids with respect to first token in the seq
        # e.g. token_ids [[1, 3, 5, 7],..[]], forward_targets=[[3,5,7],..]
        forward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]

        if self.bidirectional:
            backward_targets = torch.zeros_like(token_ids)
            backward_targets[:, 1:] = token_ids[:, 0:-1]
        else:
            backward_targets = None

        # add dropout
        contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

        # compute softmax loss
        try:
            forward_loss, backward_loss = self._compute_loss(
                contextual_embeddings_with_dropout, forward_targets, backward_targets
            )
        except IndexError:
            raise IndexError(
                "Word token out of vocabulary boundaries, please check your vocab is correctly set"
                " or created before starting training."
            )

        num_targets = torch.sum((forward_targets > 0).long())

        if num_targets > 0:
            if self.bidirectional:
                average_loss = (
                    0.5 * (forward_loss + backward_loss) / num_targets.float()
                )
            else:
                average_loss = forward_loss / num_targets.float()
        else:
            average_loss = torch.tensor(0.0)

        for metric in self.metrics.values():
            # Perplexity needs the value to be on the cpu
            metric(average_loss.to("cpu"))

        return TaskOutput(
            logits=None,
            probs=None,
            loss=average_loss,
            **{"lm_embeddings": contextual_embeddings, "mask": mask},
        )

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

    def _compute_loss(
        self,
        lm_embeddings: torch.Tensor,
        forward_targets: torch.Tensor,
        backward_targets: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If bidirectional, lm_embeddings is shape (batch_size, timesteps, dim * 2)
        # If unidirectional, lm_embeddings is shape (batch_size, timesteps, dim)
        # forward_targets, backward_targets (None in the unidirectional case) are
        # shape (batch_size, timesteps) masked with 0
        if self.bidirectional:
            forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
            backward_loss = self._loss_helper(backward_embeddings, backward_targets)
        else:
            forward_embeddings = lm_embeddings
            backward_loss = None

        forward_loss = self._loss_helper(forward_embeddings, forward_targets)
        return forward_loss, backward_loss

    def _loss_helper(
        self,
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
    """Configuration for language model head components"""

    pass
