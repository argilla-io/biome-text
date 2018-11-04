import os
import tempfile
import unittest

from allennlp.common import Params

from recognai.commands.train.train import train_model_from_file
from recognai.models import SequencePairClassifier
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

DEFINITION_TRAIN = os.path.join(TEST_RESOURCES, 'resources/definitions/train/train_sequence_pair_classifier.json')
TRAINER_PATH = os.path.join(TEST_RESOURCES, 'resources/definitions/train/trainer.json')


class TrainSeqListClassifierTest(DaskSupportTest):
    def test_train_model_from_file(self):
        output_dir = tempfile.mkdtemp()
        serialization_dir = os.path.join(output_dir, "test")
        _ = train_model_from_file(parameter_filename=DEFINITION_TRAIN,
                                  serialization_dir=serialization_dir,
                                  trainer_path=TRAINER_PATH,
                                  vocab_path=output_dir)

        model = SequencePairClassifier.from_params(vocab=None, params=Params({
            "model_location": os.path.join(serialization_dir, 'model.tar.gz')
        }))

        print (serialization_dir)
        assert model
        assert isinstance(model, SequencePairClassifier)
