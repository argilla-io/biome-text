import logging
import unittest

logging.basicConfig(level=logging.DEBUG)

import os
import tempfile

from allennlp.common import Params

from biome.text.commands.learn import learn
from biome.text.models import SequencePairClassifier
from tests.test_support import DaskSupportTest
from tests.test_context import TEST_RESOURCES

DEFINITION_TRAIN = os.path.join(
    TEST_RESOURCES, "resources/definitions/train/train_sequence_pair_classifier.json"
)
TRAINER_PATH = os.path.join(TEST_RESOURCES, "resources/definitions/train/trainer.json")
TRAIN_DATA_PATH = os.path.join(
    TEST_RESOURCES, "resources/definitions/train/train.data.json"
)
VALIDATION_DATA_PATH = os.path.join(
    TEST_RESOURCES, "resources/definitions/train/validation.data.json"
)


# TODO @dvilasuero check this test when SequenceClassifier refactor is done
@unittest.skip(reason="SequenceClassifier refactor must be finish")
class TrainSeqListClassifierTest(DaskSupportTest):
    @staticmethod
    def test_train_model_from_file():
        output_dir = tempfile.mkdtemp()
        serialization_dir = os.path.join(output_dir, "test")
        _ = learn(
            model_spec=DEFINITION_TRAIN,
            output=serialization_dir,
            train_cfg=TRAIN_DATA_PATH,
            validation_cfg=VALIDATION_DATA_PATH,
            trainer_path=TRAINER_PATH,
        )

        model = SequencePairClassifier.from_params(
            params=Params(
                {"model_location": os.path.join(serialization_dir, "model.tar.gz")}
            ),
            vocab=None,
        )

        assert model
        assert isinstance(model, SequencePairClassifier)
