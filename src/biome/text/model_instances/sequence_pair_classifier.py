from typing import Union, List

from biome.text import BaseModelInstance


class SequencePairClassifier(BaseModelInstance):
    def predict(
        self, record1: Union[str, List[str], dict], record2: Union[str, List[str], dict]
    ):
        instance = self.pipeline.text_to_instance(record1=record1, record2=record2)
        return self.architecture.forward_on_instance(instance)
