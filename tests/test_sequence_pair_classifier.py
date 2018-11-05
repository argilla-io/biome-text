import logging

logging.basicConfig(level=logging.DEBUG)

import os
import tempfile

from allennlp.common import Params

from biome.commands.learn.learn import train_model_from_file
from biome.models import SequencePairClassifier
from tests.test_support import DaskSupportTest
from tests.test_context import TEST_RESOURCES

DEFINITION_TRAIN = os.path.join(TEST_RESOURCES, 'resources/definitions/train/train_sequence_pair_classifier.json')
TRAINER_PATH = os.path.join(TEST_RESOURCES, 'resources/definitions/train/trainer.json')
TRAIN_DATA_PATH = os.path.join(TEST_RESOURCES, 'resources/definitions/train/train.data.json')
VALIDATION_DATA_PATH = os.path.join(TEST_RESOURCES, 'resources/definitions/train/validation.data.json')


class TrainSeqListClassifierTest(DaskSupportTest):
    def test_train_model_from_file(self):
        output_dir = tempfile.mkdtemp()
        serialization_dir = os.path.join(output_dir, "test")
        _ = train_model_from_file(parameter_filename=DEFINITION_TRAIN,
                                  train_cfg=TRAIN_DATA_PATH,
                                  validation_cfg=VALIDATION_DATA_PATH,
                                  serialization_dir=serialization_dir,
                                  trainer_path=TRAINER_PATH,
                                  vocab_path=output_dir)

        model = SequencePairClassifier.from_params(vocab=None, params=Params({
            "model_location": os.path.join(serialization_dir, 'model.tar.gz')
        }))

        assert model
        assert isinstance(model, SequencePairClassifier)
