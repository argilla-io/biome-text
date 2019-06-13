import os
import tempfile
from typing import Iterable

import yaml
from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, LabelField
from biome.allennlp.dataset_readers import SequenceClassifierDatasetReader
from tests.test_context import TEST_RESOURCES, create_temp_configuration

from tests.test_support import DaskSupportTest

CSV_PATH = os.path.abspath(
    os.path.join(TEST_RESOURCES, "resources/data/dataset_source.csv")
)
NO_HEADER_CSV_PATH = os.path.abspath(
    os.path.join(TEST_RESOURCES, "resources/data/no.header.dataset_source.csv")
)
JSON_PATH = os.path.abspath(
    os.path.join(TEST_RESOURCES, "resources/data/dataset_source.jsonl")
)

TOKENS_FIELD = "tokens"
LABEL_FIELD = "label"

reader = SequenceClassifierDatasetReader()


class SequenceClassifierDatasetReaderTest(DaskSupportTest):
    def test_dataset_reader_registration(self):
        dataset_reader = DatasetReader.by_name("sequence_classifier")
        self.assertEqual(SequenceClassifierDatasetReader, dataset_reader)

    def test_read_input_csv(self):
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

        yaml_config = os.path.join(
            TEST_RESOURCES, "resources/datasets/biome.csv.spec.yml"
        )
        dataset = reader.read(yaml_config)
        self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)
        self._check_dataset(dataset, expected_length, expected_inputs, expected_labels)

    def test_read_input_csv_with_no_header(self):

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

        datasource_cfg = dict(
            format="csv",
            path=NO_HEADER_CSV_PATH,
            sep=",",
            header=None,
            forward=dict(tokens=[0], target=dict(gold_label=1)),
        )

        with tempfile.NamedTemporaryFile("w") as cfg_file:
            yaml.dump(datasource_cfg, cfg_file)
            dataset = reader.read(cfg_file.name)

            self._check_dataset(
                dataset, expected_length, expected_inputs, expected_labels
            )

    def test_reader_csv_with_mappings(self):
        expected_length = 9
        expected_inputs = [
            "44",
            "blue",
            "-",
            "collar",
            "married",
            "53",
            "technician",
            "married",
            "39",
            "services",
            "married",
            "55",
            "retired",
            "married",
            "37",
            "married",
            "36",
            "admin",
            "married",
            "28",
            "management",
            "single",
            "30",
            "management",
            "divorced",
            "39",
            "divorced",
            ".",
        ]

        datasource_cfg = dict(
            format="csv",
            path=CSV_PATH,
            sep=",",
            forward=dict(
                tokens=["age", "job", "marital"], target=dict(gold_label="housing")
            ),
        )
        with tempfile.NamedTemporaryFile("w") as cfg_file:
            yaml.dump(datasource_cfg, cfg_file)
            dataset = reader.read(cfg_file.name)

            self._check_dataset(
                dataset, expected_length, expected_inputs, ["yes", "no"]
            )

    def test_reader_csv_with_leading_and_trailing_spaces_in_header(self):
        expected_length = 3
        expected_inputs = ["1", "2", "3"]
        local_data_path = os.path.join(
            TEST_RESOURCES,
            "resources/data/french_customer_data_clean_3_missing_label.csv",
        )

        datasource_cfg = dict(
            format="csv",
            path=local_data_path,
            sep=";",
            forward=dict(tokens=["dataset id"], target=dict(gold_label="dataset id")),
        )
        with tempfile.NamedTemporaryFile("w") as cfg_file:
            yaml.dump(datasource_cfg, cfg_file)
            dataset = reader.read(cfg_file.name)

            self._check_dataset(
                dataset, expected_length, expected_inputs, ["1", "2", "3"]
            )

    def test_reader_csv_with_leading_and_trailing_spaces_in_examples(self):
        expectedDatasetLength = 2
        expected_inputs = ["Dufils", "Anne", "Pierre", "Jousseaume", "Thierry"]
        expected_labels = ["11205 - Nurses: Private, Ho", "person"]
        local_data_path = os.path.join(
            TEST_RESOURCES, "resources/data/french_customer_data_clean_2.csv"
        )

        datasource_cfg = dict(
            format="csv",
            path=local_data_path,
            sep=";",
            forward={
                "tokens": ["name"],
                "target": {"gold_label": "category of institution"},
            },
        )
        with tempfile.NamedTemporaryFile("w") as cfg_file:
            yaml.dump(datasource_cfg, cfg_file)
            dataset = reader.read(cfg_file.name)

            self._check_dataset(
                dataset, expectedDatasetLength, expected_inputs, expected_labels
            )

    def test_reader_csv_with_missing_label_and_partial_mappings(self):
        """
        When label value is misssing on some examples, this fails as no LabelType is added to allen reader.
        There are several options:
            1 - Skip examples with missing labels. It might make sense or not to add an extra class for handling this.
            2 - Map missing labels to a special class: None, or other.. This could be done with transformations.
        A related issue is: when mapping label values, we might be interested in mapping only some values, leaving the others unmapped.
        We can call this "partial mapping".
        This is useful when we do not know in advance all labels, or we don't want to map them one by one.
        We tackle option 2 here and assume we have a solution for partial mappings.
        """

        expected_length = 3
        expected_inputs = [
            "Colin",
            "Revol",
            "Roger",
            "Dufils",
            "Anne",
            "Pierre",
            "Jousseaume",
            "Thierry",
        ]
        # 'None' is a custom partial mapping for missing labels
        expected_labels = ["11205 - Nurses: Private, Ho", "person", "NOLABEL"]
        local_data_path = os.path.join(
            TEST_RESOURCES,
            "resources/data/french_customer_data_clean_3_missing_label.csv",
        )

        datasource_cfg = dict(
            format="csv",
            path=local_data_path,
            sep=";",
            forward={
                "tokens": ["name"],
                "target": {
                    "gold_label": "category of institution",
                    "use_missing_label": "NOLABEL",
                    "values_mapping": {"None": "NOLABEL"},
                },
            },
        )
        with tempfile.NamedTemporaryFile("w") as cfg_file:
            yaml.dump(datasource_cfg, cfg_file)
            dataset = reader.read(cfg_file.name)

            self._check_dataset(
                dataset, expected_length, expected_inputs, expected_labels
            )

    def test_reader_csv_uk_data(self):
        expected_length = 9
        expected_inputs = None
        expected_labels = [
            "None ",
            "None",
            "None",
            "Assembly Rooms Edinburgh",
            "Fortress Technology (europe) Ltd",
            "None",
            "None",
            "Scott David",
            "  ",
            "None",
            "Corby Borough Council",
        ]
        local_data_path = os.path.abspath(
            os.path.join(TEST_RESOURCES, "resources/data/uk_customer_data_10.csv")
        )

        datasource_cfg = dict(
            format="csv",
            path=local_data_path,
            sep=";",
            forward={
                "tokens": ["name"],
                "target": {
                    "gold_label": "organisation name",
                    "use_missing_label": "None",
                },
            },
        )
        with tempfile.NamedTemporaryFile("w") as cfg_file:
            yaml.dump(datasource_cfg, cfg_file)
            dataset = reader.read(cfg_file.name)

            self._check_dataset(
                dataset, expected_length, expected_inputs, expected_labels
            )

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

        instances = list(dataset)
        self.assertEqual(expected_length, len(instances))
        for example in instances:
            self.assertTrue(
                TOKENS_FIELD in example.fields,
                f"{TOKENS_FIELD} not found in {example.fields}",
            )
            self.assertTrue(
                LABEL_FIELD in example.fields,
                f"{LABEL_FIELD} not found in {example.fields}",
            )
            check_text_field(example.fields[TOKENS_FIELD], expected_inputs)
            check_label_field(example.fields[LABEL_FIELD], expected_labels)

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


if __name__ == "__main__":
    unittest.main()
