import os
import unittest

from biome.allennlp.dataset_readers.biluo_tagger_ds_reader import (
    BiluoSequenceTaggerDatasetReader,
)
from tests import TEST_RESOURCES


class BiluoSequenceTaggerDatasetReaderTest(unittest.TestCase):

    reader = BiluoSequenceTaggerDatasetReader(lazy=False)
    data_path = os.path.join(TEST_RESOURCES, "data/biluo.data.json")
    expected_dataset_size = 1

    def test_read_biluo_dataset(self):
        instances = self.reader.read(self.data_path)
        instances = list(instances)

        self.assertEqual(self.expected_dataset_size, len(instances))

        for instance in instances:
            self.assertTrue("tokens" in instance.fields)
            self.assertTrue("tags" in instance.fields)
