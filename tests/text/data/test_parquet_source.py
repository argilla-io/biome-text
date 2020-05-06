import os

from biome.text.data import DataSource
from tests import DaskSupportTest, RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


class ParquetDataSourceTest(DaskSupportTest):
    def test_read_parquet(self):
        file_path = os.path.join(FILES_PATH, "test.parquet")
        ds = DataSource(format="parquet", path=file_path)

        df = ds.to_dataframe().compute()
        self.assertTrue("reviewerID" in df.columns)
        self.assertTrue("path" in df.columns)
