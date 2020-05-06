import os

from biome.text.data import DataSource
from tests import DaskSupportTest, RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


class JsonDatasourceTest(DaskSupportTest):
    def test_read_json(self):
        file_path = os.path.join(FILES_PATH, "dataset_source.jsonl")

        datasource = DataSource(format="json", path=file_path)
        data_frame = datasource.to_dataframe().compute()

        assert len(data_frame) > 0
        self.assertTrue("path" in data_frame.columns)

    def test_flatten_json(self):
        file_path = os.path.join(FILES_PATH, "to-be-flattened.jsonl")
        ds = DataSource(format="json", flatten=True, path=file_path)
        df = ds.to_dataframe().compute()

        for c in ["persons.*.lastName", "persons.*.name"]:
            self.assertIn(c, df.columns, f"Expected {c} as column name")

    def test_flatten_nested_list(self):
        file_path = os.path.join(FILES_PATH, "nested-list.jsonl")

        ds = DataSource(format="json", flatten=True, path=file_path)
        df = ds.to_dataframe().compute()

        for c in ["classification.*.origin.*.key", "classification.*.origin.*.source"]:
            self.assertIn(c, df.columns, f"Expected {c} as data column")
