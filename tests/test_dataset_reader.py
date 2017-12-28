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
json_dataset = os.path.join(TEST_RESOURCES, 'resources/dataset_source.jsonl')


class DatasetReaderTest(unittest.TestCase):
    def test_dataset_reader_registration(self):
        dataset_reader = DatasetReader.by_name('classification_dataset_reader')
        self.assertEquals(ClassificationDatasetReader, dataset_reader)

    def test_read_input_csv(self):
        expectedDatasetLength = 9
        expectedLabels = ['blue-collar', 'technician', 'management', 'services', 'retired', 'admin.']
        expectedInputs = ['44', '53', '28', '39', '55', '30', '37', '36']

        json_config = os.path.join(TEST_RESOURCES, 'resources/datasetReaderConfig.json')
        with open(json_config) as json_file:
            params = json.loads(json_file.read())
            reader = ClassificationDatasetReader.from_params(params=Params(params))
            dataset = reader.read(dataset_path)

            self._check_dataset(dataset, expectedDatasetLength, expectedInputs, expectedLabels)

    def test_reader_csv_with_mappings(self):
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
            [self.assertTrue(token.text in expectedInputs, msg="expected [%s] in input" % token.text) for token in
             textField.tokens]

        def check_label_field(labelField: LabelField, expectedLabels: Iterable):
            self.assertTrue(labelField.label in expectedLabels, msg="expected [%s] in labels" % labelField.label)

        self.assertEqual(expectedDatasetLength, len(dataset.instances))
        for example in dataset.instances:
            self.assertTrue(ClassificationDatasetReader._TOKENS_FIELD in example.fields)
            self.assertTrue(ClassificationDatasetReader._LABEL_FIELD in example.fields)

            check_text_field(example.fields[ClassificationDatasetReader._TOKENS_FIELD], expectedInputs)
            check_label_field(example.fields[ClassificationDatasetReader._LABEL_FIELD], expectedLabels)

    def test_read_json(self):
        with open(os.path.join(TEST_RESOURCES, 'resources/readerFromJson.json')) as json_file:
            params = json.loads(json_file.read())
            reader = ClassificationDatasetReader.from_params(params=Params(params))

            dataset = reader.read(json_dataset)
            for example in dataset.instances:
                print(example.__dict__)


if __name__ == "__main__":
    unittest.main()
