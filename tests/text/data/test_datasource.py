import os

import pytest

from biome.data.sources import DataSource
from tests import DaskSupportTest, RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


class DataSourceTest(DaskSupportTest):
    def test_wrong_format(self):
        with pytest.raises(TypeError):
            DataSource(format="not-found")
        # New format
        with pytest.raises(TypeError):
            DataSource(source="not-found")

    def test_add_mock_format(self):
        def ds_parser(*args, **kwargs):
            from dask import dataframe as ddf
            import pandas as pd

            return ddf.from_pandas(
                pd.DataFrame([i for i in range(0, 100)]), npartitions=1
            )

        DataSource.add_supported_format("new-format", ds_parser)

        for ds in [DataSource(format="new-format"), DataSource(source="new-format")]:
            self.assertFalse(ds.to_dataframe().columns is None)

    def test_to_mapped(self):
        the_mapping = {"label": "overall", "tokens": "summary"}

        for ds in [
            DataSource(
                format="json",
                mapping=the_mapping,
                path=os.path.join(FILES_PATH, "dataset_source.jsonl"),
            ),
            DataSource(
                source=os.path.join(FILES_PATH, "dataset_source.jsonl"),
                mapping=the_mapping,
            ),
        ]:
            df = ds.to_mapped_dataframe()

            self.assertIn("label", df.columns)
            self.assertIn("tokens", df.columns)

            bag = ds.to_mapped_bag().take(1)[0]

            self.assertIn("label", bag)
            self.assertIn("tokens", bag)

    def test_no_mapping(self):

        ds = DataSource(
            format="json", path=os.path.join(FILES_PATH, "dataset_source.jsonl")
        )
        with pytest.raises(ValueError):
            ds.to_mapped_dataframe()

    def test_load_multiple_formats(self):
        files = [
            os.path.join(FILES_PATH, "dataset_source.jsonl"),
            os.path.join(FILES_PATH, "dataset_source.csv"),
        ]
        with pytest.raises(TypeError):
            DataSource(source=files)

    def test_override_format(self):
        with pytest.raises(TypeError):
            DataSource(source=os.path.join(FILES_PATH, "*.jsonl"), format="not-found")
