import os
import tempfile

from tests import TESTS_BASEPATH
from tests.test_support import DaskSupportTest

FILES_PATH = os.path.join(TESTS_BASEPATH, "resources/files")

from biome.data.sources import DataSource
from biome.data.sinks.helpers import store_dataset


class ExcelDatasourceTest(DaskSupportTest):
    def test_read_and_write_excel(self):
        def drop_keys(data, keys):
            return {k: v for k, v in data.items() if k not in keys}

        file_path = os.path.join(FILES_PATH, "test.xlsx")

        datasource = DataSource(format="xlsx", path=file_path)

        tmpfile = tempfile.mkdtemp()
        store_dataset(datasource.to_bag(), dict(path=tmpfile))

        stored_dataset = DataSource(format="json", path=os.path.join(tmpfile, "*.part"))

        stored = stored_dataset.to_bag().compute()
        read = datasource.to_bag().compute()

        self.assertEqual(len(stored), len(read))

        variable_keys = ["resource", "id"]
        [
            self.assertEqual(drop_keys(a, variable_keys), drop_keys(a, variable_keys))
            for a, b in zip(read, stored)
        ]
