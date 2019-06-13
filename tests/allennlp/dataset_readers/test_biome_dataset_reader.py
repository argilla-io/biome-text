import os
from typing import Iterable

from allennlp.data.fields import TextField, LabelField
from biome.allennlp.dataset_readers import SequenceClassifierDatasetReader

from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

TOKENS_FIELD = "tokens"
LABEL_FIELD = "label"

reader = SequenceClassifierDatasetReader()


class BiomeDatasetReaderTest(DaskSupportTest):
    def test_read_input_csv_multi_file(self):
        expected_length = 18  # Two times the same file
        expected_labels = [
            "blue-collar",
            "technician",
            "management",
            "services",
            "retired",
            "admin.",
        ]
        expected_inputs = ["44", "53", "28", "39", "55", "30", "37", "36"]

        yaml_config = os.path.join(
            TEST_RESOURCES, "resources/datasets/biome.csv.multi.file.spec.yml"
        )
        dataset = reader.read(yaml_config)
        self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)

    def test_read_input_json(self):

        yaml_config = os.path.join(
            TEST_RESOURCES, "resources/datasets/biome.json.spec.yml"
        )
        dataset = list(reader.read(yaml_config))

        assert len(dataset) == 5
        for example in dataset:
            print(example.fields)

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

        lInstances = list(dataset)
        self.assertEqual(expected_length, len(lInstances))
        for example in lInstances:
            self.assertTrue(TOKENS_FIELD in example.fields)
            self.assertTrue(LABEL_FIELD in example.fields)
            check_text_field(example.fields[TOKENS_FIELD], expected_inputs)
            check_label_field(example.fields[LABEL_FIELD], expected_labels)
