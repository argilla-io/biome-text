import os

from biome.text.models import SequencePairClassifier
from tests.test_context import TEST_RESOURCES
from tests.text.models.base_classifier import BasePairClassifierTest

BASE_CONFIG_PATH = os.path.join(
    TEST_RESOURCES, "resources/models/sequence_pair_classifier"
)


class SequencePairClassifierTest(BasePairClassifierTest):
    base_config = BASE_CONFIG_PATH

    def test_model_workflow(self):
        self.check_train(SequencePairClassifier)
        self.check_predictor()
        # self.check_serve()
        self.check_explore()
