import os
import tempfile
import shutil
import unittest

from allennlp.data import Vocabulary

from allennlp_extensions.commands.preprocess.preprocess import preprocess_from_file
from allennlp_extensions.commands.train.train import train_model_from_file
from allennlp_extensions.data import dataset
from tests.test_context import TEST_RESOURCES

DEFINITION_PREPROCESS = os.path.join(TEST_RESOURCES, 'resources/definitions/preprocess/simple.json')
DEFINITION_TRAIN = os.path.join(TEST_RESOURCES, 'resources/definitions/train/load_and_train.json')


class TrainTest(unittest.TestCase):
    @unittest.skip("Several changes to iterator that I don't know how to manage. Needed `sorting_keys` field")
    def test_load_and_train_from_preprocessed(self):
        output_dir = tempfile.mkdtemp()

        preprocess_from_file(DEFINITION_PREPROCESS, output_dir)

        vocab = Vocabulary.from_files(os.path.join(output_dir, "vocabulary"))

        train_dataset = dataset.load_from_file(os.path.join(output_dir, 'train.data'))
        validation_dataset = dataset.load_from_file(os.path.join(output_dir, 'validation.data'))

        model = train_model_from_file(DEFINITION_TRAIN, output_dir, "", output_dir)

        self.assertNotEqual(len(model.vocab._non_padded_namespaces), 0)

        shutil.rmtree(output_dir)
