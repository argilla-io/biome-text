import json
import os
import tempfile

from tests import TESTS_BASEPATH
from tests.test_support import DaskSupportTest

FILES_PATH = os.path.join(TESTS_BASEPATH, "resources/files")

from biome.data.sources import DataSource
from biome.data.sinks.helpers import store_dataset


class CsvDatasourceTest(DaskSupportTest):
    def test_read_and_write_csv(self):
        file_path = os.path.join(FILES_PATH, "dataset_source.csv")

        datasource = DataSource(format="csv", path=file_path)

        tmpfile = tempfile.mkdtemp()
        store_dataset(datasource.to_bag(), dict(path=tmpfile))

        stored_dataset = DataSource(format="json", path=os.path.join(tmpfile, "*.part"))

        stored = stored_dataset.to_bag().compute()
        read = datasource.to_bag().compute()

        assert len(stored) == len(read)

        drop_keys = lambda data, keys: {k: v for k, v in data.items() if k not in keys}
        variable_keys = ["resource", "id"]

        for a, b in zip(read, stored):
            assert drop_keys(a, variable_keys) == drop_keys(a, variable_keys)
