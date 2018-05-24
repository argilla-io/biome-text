import os
import tempfile
import shutil
import unittest

from allennlp.data import Vocabulary

from recognai.commands.preprocess.preprocess import preprocess_from_file
from recognai.data import dataset
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

DEFINITION = os.path.join(TEST_RESOURCES, 'resources/definitions/preprocess/simple.json')


class DatasetReaderTest(DaskSupportTest):
    @unittest.skip
    def test_load_generated_datasets_OK(self):
        output_dir = tempfile.mkdtemp()

        preprocess_from_file(DEFINITION, output_dir)

        vocab = Vocabulary.from_files(os.path.join(output_dir, "vocabulary"))

        self.assertTrue(vocab.get_vocab_size("tokens") > 0)

        shutil.rmtree(output_dir)
