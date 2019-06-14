import os
from typing import Iterable

from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, LabelField
from biome.allennlp.dataset_readers import SequenceClassifierDatasetReader
from tests.test_context import TEST_RESOURCES, create_temp_configuration

from tests.test_support import DaskSupportTest

CSV_PATH = os.path.join(TEST_RESOURCES, "resources/data/dataset_source.csv")
JSON_PATH = os.path.join(TEST_RESOURCES, "resources/data/dataset_source.jsonl")
DEFINITIONS_PATH = os.path.join(TEST_RESOURCES, "resources/dataset_readers/definitions")

TOKENS_FIELD = "tokens"
LABEL_FIELD = "label"

reader = SequenceClassifierDatasetReader()


class ParallelDatasetReaderTest(DaskSupportTest):
    def test_dataset_reader_registration(self):
        dataset_reader = DatasetReader.by_name("sequence_classifier")
        self.assertEqual(SequenceClassifierDatasetReader, dataset_reader)

    def test_read_csv(self):
        expected_length = 9
        expected_labels = [
            "blue-collar",
            "technician",
            "management",
            "services",
            "retired",
            "admin.",
        ]
        expected_inputs = ["44", "53", "28", "39", "55", "30", "37", "36"]

        dataset = reader.read(
            create_temp_configuration(
                {
                    "path": CSV_PATH,
                    "format": "csv",
                    "forward": {"tokens": ["age"], "target": {"gold_label": "job"}},
                }
            )
        )

        self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)

    def test_read_json(self):
        dataset = reader.read(
            create_temp_configuration(
                {
                    "format": "json",
                    "path": JSON_PATH,
                    "forward": {
                        "tokens": ["reviewText"],
                        "target": {"gold_label": "overall"},
                    },
                }
            )
        )

        for example in dataset:
            print(example.__dict__)

    def _check_dataset(
        self,
        dataset,
        expected_length: int,
        expected_inputs: Iterable,
        expected_labels: Iterable,
    ):
        def check_text_field(textField: TextField, expected_inputs: Iterable):
            if expected_inputs:
                [
                    self.assertTrue(
                        token.text in expected_inputs,
                        msg="expected [%s] in input" % token.text,
                    )
                    for token in textField.tokens
                ]

        def check_label_field(labelField: LabelField, expected_labels: Iterable):
            self.assertTrue(
                labelField.label in expected_labels,
                msg="expected [%s] in labels" % labelField.label,
            )

        materializedInstances = list(dataset)
        self.assertEqual(expected_length, len(materializedInstances))
        for example in materializedInstances:
            self.assertTrue(TOKENS_FIELD in example.fields)
            self.assertTrue(LABEL_FIELD in example.fields)
            check_text_field(example.fields[TOKENS_FIELD], expected_inputs)
            check_label_field(example.fields[LABEL_FIELD], expected_labels)
