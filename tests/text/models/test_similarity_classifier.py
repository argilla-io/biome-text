import logging
import os

from biome.text.models import SimilarityClassifier
from tests.test_context import TEST_RESOURCES
from tests.text.models.base_classifier import BasePairClassifierTest

logging.basicConfig(level=logging.DEBUG)

BASE_CONFIG_PATH = os.path.join(
    TEST_RESOURCES, "resources/models/similarity_classifier"
)


class SimilarityClassifierTest(BasePairClassifierTest):
    base_config = BASE_CONFIG_PATH

    def test_model_workflow(self):
        self.check_train(SimilarityClassifier)
        self.check_explore()
        self.check_serve()
        self.check_predictor()
