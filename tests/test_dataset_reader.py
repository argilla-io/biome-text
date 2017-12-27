import unittest
import os
import json

from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, LabelField
from typing import Iterable

from allennlp_extensions.data.dataset_readers.classification_dataset_reader import ClassificationDatasetReader

from tests.test_context import TEST_RESOURCES

dataset_path = os.path.join(TEST_RESOURCES, 'resources/dataset_source.csv')


class DatasetReaderTest(unittest.TestCase):
    def test_dataset_reader_registration(self):
        dataset_reader = DatasetReader.by_name('classification_dataset_reader')
        self.assertEquals(ClassificationDatasetReader, dataset_reader)

    def test_read_input_config(self):
        expectedDatasetLength = 9
        expectedLabels = ['blue-collar', 'technician', 'management', 'services', 'retired', 'admin.']
        expectedInputs = ['44', '53', '28', '39', '55', '30', '37', '36']

        json_config = os.path.join(TEST_RESOURCES, 'resources/datasetReaderConfig.json')
        with open(json_config) as json_file:
            params = json.loads(json_file.read())
            reader = ClassificationDatasetReader.from_params(params=Params(params))
            dataset = reader.read(dataset_path)

            self._check_dataset(dataset, expectedDatasetLength, expectedInputs, expectedLabels)

    def test_reader_with_mappings(self):
        expectedDatasetLength = 9
        expectedInputs = ['44', 'blue', '-', 'collar', 'married', '53', 'technician', 'married', '39', 'services',
                          'married',
                          '55', 'retired', 'married', '37', 'married', '36', 'admin', 'married', '28', 'management',
                          'single', '30', 'management', 'divorced', '39', 'divorced', '.']

        with open(os.path.join(TEST_RESOURCES, 'resources/readerWithMappings.json')) as json_file:
            params = json.loads(json_file.read())
            reader = ClassificationDatasetReader.from_params(params=Params(params))
            dataset = reader.read(dataset_path)

            self._check_dataset(dataset, expectedDatasetLength, expectedInputs, ['NOT_AT_ALL', 'OF_COURSE'])

    def _check_dataset(self, dataset, expectedDatasetLength: int, expectedInputs: Iterable, expectedLabels: Iterable):
        def check_text_field(textField: TextField, expectedInputs: Iterable):
            [self.assertTrue(token.text in expectedInputs, msg="expected %s in input" % token.text) for token in
             textField.tokens]

        def check_label_field(labelField: LabelField, expectedLabels: Iterable):
            self.assertTrue(labelField.label in expectedLabels, msg="expected %s labels" % labelField.label)

        self.assertEqual(expectedDatasetLength, len(dataset.instances))
        for example in dataset.instances:
            self.assertTrue('tokens' in example.fields)
            self.assertTrue('label' in example.fields)

            check_text_field(example.fields['tokens'], expectedInputs)
            check_label_field(example.fields['label'], expectedLabels)


if __name__ == "__main__":
    unittest.main()
