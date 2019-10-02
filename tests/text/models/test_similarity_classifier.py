import logging
import os

from biome.text.models import SimilarityClassifier
from tests.test_context import TEST_RESOURCES
from tests.text.models.test_sequence_pair_classifier import SequencePairClassifierTest

logging.basicConfig(level=logging.DEBUG)

BASE_CONFIG_PATH = os.path.join(
    TEST_RESOURCES, "resources/definitions/similarity_classifier"
)


class SimilarityClassifierTest(SequencePairClassifierTest):
    name = "similarity_classifier"

    model_path = os.path.join(BASE_CONFIG_PATH, "model.yml")
    trainer_path = os.path.join(BASE_CONFIG_PATH, "trainer.yml")
    training_data = os.path.join(BASE_CONFIG_PATH, "train.data.yml")
    validation_data = os.path.join(BASE_CONFIG_PATH, "validation.data.yml")

    def test_model_workflow(self):
        self.check_train(SimilarityClassifier)
        self.check_predict()
        self.check_serve()
        self.check_predictor()
