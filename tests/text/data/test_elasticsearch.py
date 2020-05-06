import os

import pytest
from dask.dataframe import DataFrame

from biome.text.api_new.data.readers import ElasticsearchDataFrameReader
from tests.test_support import DaskSupportTest

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "test-index")
ES_DOC = os.getenv("ES_DOC", "_doc")


class ElasticsearchReaderTest(DaskSupportTest):
    @staticmethod
    def _load_data_to_elasticsearch(data, host: str, index: str, doc: str):
        from elasticsearch import Elasticsearch
        from elasticsearch import helpers

        client = Elasticsearch(hosts=host, http_compress=True)
        client.indices.delete(index, ignore_unavailable=True)

        def _generator(data):
            for document in data:
                yield {"_index": index, "_type": doc, "_source": document}

        helpers.bulk(client, _generator(data))
        del client

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_load_data(self):

        self._load_data_to_elasticsearch(
            [dict(a=i, b=f"this is {i}") for i in range(1, 5000)],
            host=ES_HOST,
            index=ES_INDEX,
            doc=ES_DOC,
        )

        es_index = ElasticsearchDataFrameReader.read(
            source=ElasticsearchDataFrameReader.SOURCE_TYPE,
            es_host=ES_HOST,
            index=ES_INDEX,
            doc_type=ES_DOC,
        )

        self.assertTrue(
            isinstance(es_index, DataFrame),
            f"elasticsearch datasource is not a dataframe :{type(es_index)}",
        )
        print(es_index.columns)
        self.assertTrue(
            "_id" not in es_index.columns and "id" in es_index.columns,
            "Expected renamed elasticsearch id",
        )
