import os

from biome.text.api_new.data import DataSource
from tests import DaskSupportTest, RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


class ExcelDataSourceTest(DaskSupportTest):
    def test_read_excel(self):
        file_path = os.path.join(FILES_PATH, "test.xlsx")

        datasource = DataSource(format="xlsx", path=file_path)
        data_frame = datasource.to_dataframe().compute()

        assert len(data_frame) > 0
        self.assertTrue("path" in data_frame.columns)
