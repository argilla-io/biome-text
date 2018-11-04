import json
import os
import unittest
from typing import Iterable, Dict

from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, LabelField

from biome.data.dataset_readers.classification_dataset_reader import ClassificationDatasetReader
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

CSV_PATH = os.path.join(TEST_RESOURCES, 'resources/data/dataset_source.csv')
JSON_PATH = os.path.join(TEST_RESOURCES, 'resources/data/dataset_source.jsonl')

TOKENS_FIELD = 'tokens'
LABEL_FIELD = 'gold_label'

CLASSIFIER_SPEC = os.path.join(TEST_RESOURCES, 'resources/dataset_readers/definitions/classifier_dataset_reader.json')
reader = ClassificationDatasetReader.from_params(params=Params.from_file(CLASSIFIER_SPEC))


def csv_data(path: str, sep: str = ';', transformations: Dict = dict()):
    return {
        'path': path,
        'sep': sep,
        'format': 'csv',
        'transformations': transformations
    }


class DatasetReaderTest(DaskSupportTest):
    def test_dataset_reader_registration(self):
        dataset_reader = DatasetReader.by_name('classification_dataset_reader')
        self.assertEquals(ClassificationDatasetReader, dataset_reader)

    def test_read_input_csv(self):
        expected_length = 9
        expected_labels = ['blue-collar', 'technician', 'management', 'services', 'retired', 'admin.']
        expected_inputs = ['44.0', '53.0', '28.0', '39.0', '55.0', '30.0', '37.0', '36.0']

        json_config = os.path.join(TEST_RESOURCES, 'resources/datasetReaderConfig.json')
        with open(json_config) as json_file:
            dataset = reader.read(csv_data(CSV_PATH, sep=',', transformations={
                "tokens": [
                    "age"
                ],
                "target": {
                    "gold_label": "job"
                }
            }))

            self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)

    def test_reader_csv_with_mappings(self):
        expected_length = 9
        expected_inputs = ['44.0', 'blue', '-', 'collar', 'married', '53.0', 'technician', 'married', '39.0',
                           'services',
                           'married',
                           '55.0', 'retired', 'married', '37.0', 'married', '36.0', 'admin', 'married', '28.0',
                           'management',
                           'single', '30.0', 'management', 'divorced', '39.0', 'divorced', '.']

        with open(os.path.join(TEST_RESOURCES, 'resources/readerWithMappings.json')) as json_file:
            dataset = reader.read(csv_data(CSV_PATH, sep=',', transformations={
                "tokens": [
                    "age",
                    "job",
                    "marital"
                ],
                "target": {
                    "gold_label": "housing",
                    "values_mapping": {
                        "yes": "OF_COURSE",
                        "no": "NOT_AT_ALL"
                    }
                }
            }))

            self._check_dataset(dataset, expected_length, expected_inputs, ['NOT_AT_ALL', 'OF_COURSE'])

    def test_reader_csv_with_leading_and_trailing_spaces_in_header(self):
        expected_length = 3
        expected_inputs = ['1.0', '2.0', '3.0']
        local_data_path = os.path.join(TEST_RESOURCES, 'resources/data/french_customer_data_clean_3_missing_label.csv')
        with open(os.path.join(TEST_RESOURCES, 'resources/datasetReaderConfigMultiwordTrailing.json')) as json_file:
            dataset = reader.read(csv_data(local_data_path, transformations={
                "tokens": [
                    "dataset id"
                ],
                "target": {
                    "gold_label": "dataset id"
                }
            }))

            self._check_dataset(dataset, expected_length, expected_inputs, ['1.0', '2.0', '3.0'])

    def test_reader_csv_with_leading_and_trailing_spaces_in_examples(self):
        expectedDatasetLength = 2
        expected_inputs = ['Dufils', 'Anne', 'Pierre', 'Jousseaume', 'Thierry']
        expected_labels = ['11205 - Nurses: Private, Ho', 'person']
        local_data_path = os.path.join(TEST_RESOURCES, 'resources/data/french_customer_data_clean_2.csv')
        with open(os.path.join(TEST_RESOURCES, 'resources/datasetReaderConfigMultiwordMappings.json')) as json_file:
            dataset = reader.read(csv_data(local_data_path, transformations={
                "tokens": [
                    "name"
                ],
                "target": {
                    "gold_label": "category of institution"
                }
            }))
            self._check_dataset(dataset, expectedDatasetLength, expected_inputs, expected_labels)

    def test_reader_csv_with_missing_label_and_partial_mappings(self):
        '''
        When label value is misssing on some examples, this fails as no LabelType is added to allen reader.
        There are several options:
            1 - Skip examples with missing labels. It might make sense or not to add an extra class for handling this.
            2 - Map missing labels to a special class: None, or other.. This could be done with transformations.
        A related issue is: when mapping label values, we might be interested in mapping only some values, leaving the others unmapped.
        We can call this "partial mapping".
        This is useful when we do not know in advance all labels, or we don't want to map them one by one.
        We tackle option 2 here and assume we have a solution for partial mappings.
        '''

        expected_length = 3
        expected_inputs = ['Colin', 'Revol', 'Roger', 'Dufils', 'Anne', 'Pierre', 'Jousseaume', 'Thierry']
        # 'None' is a custom partial mapping for missing labels
        expected_labels = ['11205 - Nurses: Private, Ho', 'person', 'NOLABEL']
        local_data_path = os.path.join(TEST_RESOURCES, 'resources/data/french_customer_data_clean_3_missing_label.csv')
        with open(os.path.join(TEST_RESOURCES,
                               'resources/datasetReaderConfigPartialMappingMissingLabel.json')) as json_file:
            dataset = reader.read(csv_data(local_data_path, transformations={
                "tokens": [
                    "name"
                ],
                "target": {
                    "gold_label": "category of institution",
                    "use_missing_label": "NOLABEL",
                    "values_mapping": {
                        "None": "NOLABEL"
                    }
                }
            }))

            self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)

    def test_reader_csv_uk_data(self):
        expected_length = 9
        expected_inputs = None
        expected_labels = ['None ', 'None', 'None', 'Assembly Rooms Edinburgh', 'Fortress Technology (europe) Ltd',
                           'None', 'None', 'Scott David', '  ', 'None', 'Corby Borough Council']
        local_data_path = os.path.join(TEST_RESOURCES, 'resources/data/uk_customer_data_10.csv')
        with open(os.path.join(TEST_RESOURCES, 'resources/datasetReaderConfigUK.json')) as json_file:
            dataset = reader.read(csv_data(local_data_path, transformations={
                "tokens": [
                    "name"
                ],
                "target": {
                    "gold_label": "organisation name",
                    "use_missing_label": "None"
                }
            }))

            self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)

    def _check_dataset(self, dataset, expected_length: int, expected_inputs: Iterable, expected_labels: Iterable):
        def check_text_field(textField: TextField, expected_inputs: Iterable):
            if expected_inputs:
                [self.assertTrue(token.text in expected_inputs, msg="expected [%s] in input" % token.text) for token in
                 textField.tokens]

        def check_label_field(labelField: LabelField, expected_labels: Iterable):
            self.assertTrue(labelField.label in expected_labels, msg="expected [%s] in labels" % labelField.label)

        lInstances = list(dataset)
        self.assertEqual(expected_length, len(lInstances))
        for example in lInstances:
            self.assertTrue(TOKENS_FIELD in example.fields)
            self.assertTrue(LABEL_FIELD in example.fields)
            check_text_field(example.fields[TOKENS_FIELD], expected_inputs)
            check_label_field(example.fields[LABEL_FIELD], expected_labels)

    def test_read_json(self):
        with open(os.path.join(TEST_RESOURCES, 'resources/readerFromJson.json')) as json_file:
            dataset = reader.read({
                'path': JSON_PATH,
                'transformations': {
                    "tokens": [
                        "reviewText"
                    ],
                    "target": {
                        "gold_label": "overall"
                    }
                }})
            for example in dataset:
                print(example.__dict__)

            if __name__ == "__main__":
                unittest.main()
