from typing import Any, Dict, List, Optional

import torch
from allennlp.data import Instance
from allennlp.models import BiMpm as AllennlpBimMpm
from allennlp.modules import BiMpmMatching, FeedForward, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from biome.text.api_new import Pipeline
from biome.text.api_new.model import Model
from biome.text.api_new.modules.heads import TaskOutput
from biome.text.api_new.modules.heads.classification.defs import ClassificationHead


class BiMpm(ClassificationHead):
    """Text classification using bimpm implementation"""

    def __init__(
        self,
        model: Model,
        labels: List[str],
        matcher_word: BiMpmMatching,
        encoder1: Seq2SeqEncoder,
        matcher_forward1: BiMpmMatching,
        matcher_backward1: BiMpmMatching,
        encoder2: Seq2SeqEncoder,
        matcher_forward2: BiMpmMatching,
        matcher_backward2: BiMpmMatching,
        aggregator: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        dropout: float = 0.1,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        accuracy: Optional[CategoricalAccuracy] = None,
    ):
        super(BiMpm, self).__init__(model, labels=labels)

        if not isinstance(self.model.encoder, PassThroughEncoder):
            raise TypeError("Cannot apply bimpm with an already defined encoder")

        self._bimpm_module = AllennlpBimMpm(
            self.model.vocab,
            text_field_embedder=self.model.featurizer.build_embedder(self.model.vocab),
            matcher_word=matcher_word,
            encoder1=encoder1,
            encoder2=encoder2,
            matcher_forward1=matcher_forward1,
            matcher_forward2=matcher_forward2,
            matcher_backward1=matcher_backward1,
            matcher_backward2=matcher_backward2,
            aggregator=aggregator,
            classifier_feedforward=classifier_feedforward,
            dropout=dropout,
            initializer=initializer,
            regularizer=regularizer,
        )

    def featurize(
        self, one: Dict[str, Any], another: Dict[str, Any], label: Optional[str] = None
    ) -> Optional[Instance]:
        one_field = self.model.featurize(one, aggregate=True)
        another_field = self.model.featurize(another, aggregate=True)
        instance = Instance(
            {"one": one_field.get("one"), "another": another_field.get("another")}
        )
        return self.add_label(instance, label)

    def forward(
        self,
        one: Dict[str, torch.Tensor],
        another: Dict[str, torch.Tensor],
        label: Optional[torch.Tensor] = None,
    ) -> TaskOutput:
        output_dict = self._bimpm_module.forward(
            premise=one, hypothesis=another, label=label
        )
        return TaskOutput(**output_dict)


BiMpm.register(overrides=True)

if __name__ == "__main__":
    pipeline = Pipeline.from_file("bimpm.yaml")
    print(pipeline.predict(one="The one", another="The other"))
