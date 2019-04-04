import os
import unittest

from biome.allennlp.dataset_readers.biluo_tagger_ds_reader import (
    BiluoSequenceTaggerDatasetReader,
)
from tests import TEST_RESOURCES


class BiluoSequenceTaggerDatasetReaderTest(unittest.TestCase):

    biluo_path = os.path.join(TEST_RESOURCES, "data/biluo.data.json")
    spans_path = os.path.join(TEST_RESOURCES, "data/spans.data.json")
    expected_dataset_size = 1

    def test_read_biluo_dataset(self):
        reader = BiluoSequenceTaggerDatasetReader(lazy=False)
        instances = reader.read(self.biluo_path)
        instances = list(instances)

        self.assertEqual(self.expected_dataset_size, len(instances))

        for instance in instances:
            self.assertTrue("tokens" in instance.fields)
            self.assertTrue("tags" in instance.fields)
            self.assertTrue("metadata" in instance.fields)

    def test_read_spans_dataset(self):
        reader = BiluoSequenceTaggerDatasetReader(lazy=False, format="spans")
        instances = reader.read(self.spans_path)
        instances = list(instances)

        self.assertEqual(self.expected_dataset_size, len(instances))

        for instance in instances:
            self.assertTrue("tokens" in instance.fields)
            self.assertTrue("tags" in instance.fields)
            self.assertTrue("metadata" in instance.fields)
