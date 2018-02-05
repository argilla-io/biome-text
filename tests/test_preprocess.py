import os
import tempfile
import shutil
import unittest

from allennlp.data import Vocabulary

from allennlp_extensions.commands.preprocess.preprocess import preprocess_from_file
from allennlp_extensions.data import dataset
from tests.test_context import TEST_RESOURCES

DEFINITION = os.path.join(TEST_RESOURCES, 'resources/definitions/preprocess/simple.json')


class DatasetReaderTest(unittest.TestCase):
    def test_load_generated_datasets_OK(self):
        output_dir = tempfile.mkdtemp()

        preprocess_from_file(DEFINITION, output_dir)

        vocab = Vocabulary.from_files(os.path.join(output_dir, "vocabulary"))

        train_dataset = dataset.load_from_file(os.path.join(output_dir, 'train.data'))
        validation_dataset = dataset.load_from_file(os.path.join(output_dir, 'validation.data'))

        self.assertNotEqual(1, len(list(train_dataset)))
        self.assertNotEqual(1, len(list(validation_dataset)))

        shutil.rmtree(output_dir)
