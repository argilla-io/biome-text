import os

import pytest

from biome.data.sources.readers import from_elasticsearch
from dask.dataframe import DataFrame

from tests.test_support import DaskSupportTest

NPARTITIONS = 4
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "test-index")


class ElasticsearchReaderTest(DaskSupportTest):
    @pytest.mark.skip(
        reason="Must be configured. Generate some mock data before running test"
    )
    def test_read_whole_index(self):
        es_index = from_elasticsearch(
            npartitions=NPARTITIONS, client_kwargs={"hosts": ES_HOST}, index=ES_INDEX
        )

        self.assertTrue(
            isinstance(es_index, DataFrame),
            f"elasticsearch datasource is not a dataframe :{type(es_index)}",
        )
        self.assertTrue(
            es_index.npartitions == NPARTITIONS, "Wrong number of partitions"
        )
        self.assertTrue("id" not in es_index.columns.values, "Expected id as index")
