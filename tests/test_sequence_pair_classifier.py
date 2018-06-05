import os
import tempfile
import unittest

from recognai.commands.train.train import train_model_from_file
from tests.test_context import TEST_RESOURCES

DEFINITION_TRAIN = os.path.join(TEST_RESOURCES, 'resources/definitions/train/train_sequence_pair_classifier.json')


class TrainSeqListClassifierTest(unittest.TestCase):
    def test_train_model_from_file(self):
        output_dir = tempfile.mkdtemp()
        model = train_model_from_file(DEFINITION_TRAIN, output_dir + "/test", "", output_dir )