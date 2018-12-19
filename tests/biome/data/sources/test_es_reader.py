import unittest

from biome.data.sources import elasticsearch

from tests.test_support import DaskSupportTest

NPARTITIONS = 4
ES_HOST = 'http://localhost:9200'
ES_INDEX = 'gourmet-food'


class ElasticsearchReaderTest(DaskSupportTest):
    @unittest.skip
    def test_read_whole_index(self):
        es_index = elasticsearch.from_elasticsearch(
            npartitions=NPARTITIONS,
            client_kwargs={'hosts': ES_HOST},
            index=ES_INDEX,
            source_only=True
        )

        self.assertTrue(es_index.npartitions == NPARTITIONS)

        for example in es_index:
            print(example)
