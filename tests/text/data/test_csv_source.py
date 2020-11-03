from biome.text.data import DataSource
import os

from tests import DaskSupportTest, RESOURCES_PATH

FILES_PATH = os.path.join(RESOURCES_PATH, "data")


class CsvDatasourceTest(DaskSupportTest):
    def test_reader_csv_with_leading_and_trailing_spaces_in_examples(self):
        ds = DataSource(
            format="csv",
            source=os.path.join(FILES_PATH, "trailing_coma_in_headers.csv"),
            sep=";",
        )
        df = ds.to_dataframe().compute()
        self.assertIn("name", df.columns)
