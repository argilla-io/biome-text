import json
import os
import unittest
from typing import Iterable

from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, LabelField

from allennlp_extensions.data.dataset_readers.classification_dataset_reader import ParallelDatasetReader, \
    TOKENS_FIELD, LABEL_FIELD
from tests.test_context import TEST_RESOURCES

CSV_PATH = os.path.join(TEST_RESOURCES, 'resources/data/dataset_source.csv')
JSON_PATH = os.path.join(TEST_RESOURCES, 'resources/data/dataset_source.jsonl')
DEFINITIONS_PATH = os.path.join(TEST_RESOURCES, 'resources/dataset_readers/definitions')


class ParallelDatasetReaderTest(unittest.TestCase):

    def test_dataset_reader_registration(self):
        dataset_reader = DatasetReader.by_name('parallel_dataset_reader')
        self.assertEquals(ParallelDatasetReader, dataset_reader)

    def test_read_csv(self):
        expected_length = 9
        expected_labels = ['blue-collar', 'technician', 'management', 'services', 'retired', 'admin.']
        expected_inputs = ['44.0', '53.0', '28.0', '39.0', '55.0', '30.0', '37.0', '36.0']

        json_config = os.path.join(TEST_RESOURCES, DEFINITIONS_PATH, 'parallel_definition.csv.json')
        with open(json_config) as json_file:
            params = json.loads(json_file.read())
            reader = ParallelDatasetReader.from_params(params=Params(params))

            dataset = reader.read(CSV_PATH)

            self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)

    def test_read_json(self):
        json_config = os.path.join(TEST_RESOURCES, DEFINITIONS_PATH, 'parallel_definition.json.json')
        with open(json_config) as json_file:
            params = json.loads(json_file.read())
            reader = ParallelDatasetReader.from_params(params=Params(params))

            dataset = reader.read(JSON_PATH)

            for example in dataset:
                print(example.__dict__)

    def _check_dataset(self, dataset, expected_length: int, expected_inputs: Iterable, expected_labels: Iterable):
        def check_text_field(textField: TextField, expected_inputs: Iterable):
            if expected_inputs:
                [self.assertTrue(token.text in expected_inputs, msg="expected [%s] in input" % token.text) for token in
                 textField.tokens]

        def check_label_field(labelField: LabelField, expected_labels: Iterable):
            self.assertTrue(labelField.label in expected_labels, msg="expected [%s] in labels" % labelField.label)

        materializedInstances = list(dataset)
        self.assertEqual(expected_length, len(materializedInstances))
        for example in materializedInstances:
            self.assertTrue(TOKENS_FIELD in example.fields)
            self.assertTrue(LABEL_FIELD in example.fields)
            check_text_field(example.fields[TOKENS_FIELD], expected_inputs)
            check_label_field(example.fields[LABEL_FIELD], expected_labels)
