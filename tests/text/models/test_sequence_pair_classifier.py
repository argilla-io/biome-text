import logging
import os
import tempfile

from allennlp.models import load_archive

from biome.text.commands.learn.learn import learn
from biome.text.models import SequencePairClassifier
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

logging.basicConfig(level=logging.DEBUG)

DEFINITION_TRAIN = os.path.join(
    TEST_RESOURCES,
    "resources/definitions/sequence_pair_classifier/train_sequence_pair_classifier.yml",
)
TRAINER_PATH = os.path.join(
    TEST_RESOURCES, "resources/definitions/sequence_pair_classifier/trainer.yml"
)
TRAIN_DATA_PATH = os.path.join(
    TEST_RESOURCES, "resources/definitions/sequence_pair_classifier/train.data.yml"
)
VALIDATION_DATA_PATH = os.path.join(
    TEST_RESOURCES, "resources/definitions/sequence_pair_classifier/validation.data.yml"
)


class SequencePairClassifierTest(DaskSupportTest):
    def test_train_model_from_file(self):
        output_dir = tempfile.mkdtemp()
        serialization_dir = os.path.join(output_dir, "test")
        _ = learn(
            model_spec=DEFINITION_TRAIN,
            output=serialization_dir,
            train_cfg=TRAIN_DATA_PATH,
            validation_cfg=VALIDATION_DATA_PATH,
            trainer_path=TRAINER_PATH,
        )

        archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"))

        self.assertTrue(archive.model is not None)
        self.assertIsInstance(archive.model, SequencePairClassifier)
