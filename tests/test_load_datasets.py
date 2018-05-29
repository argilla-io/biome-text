import os
import tempfile
import shutil
import unittest

from allennlp.data import Vocabulary

from recognai.commands.preprocess.preprocess import preprocess_from_file
from recognai.commands.train.train import train_model_from_file
from recognai.data import dataset
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

DEFINITION_PREPROCESS = os.path.join(TEST_RESOURCES, 'resources/definitions/preprocess/simple.yml')
DEFINITION_TRAIN = os.path.join(TEST_RESOURCES, 'resources/definitions/train/load_and_train.json')


class TrainTest(DaskSupportTest):
    @unittest.skip("Several changes to iterator that I don't know how to manage. Needed `sorting_keys` field")
    def test_load_and_train_from_preprocessed(self):
        output_dir = tempfile.mkdtemp()

        preprocess_from_file(DEFINITION_PREPROCESS, output_dir)
        shutil.rmtree(output_dir)
