from elasticsearch import Elasticsearch

from biome.text import helpers


class ElasticsearchConfig:
    """ Elasticsearch configuration data class"""

    def __init__(self, es_host: str, es_index: str):
        self.es_host = es_host or "localhost:9200"
        self.es_index = es_index
        self.client = Elasticsearch(
            hosts=es_host, retry_on_timeout=True, http_compress=True
        )
        self.es_doc = helpers.get_compatible_doc_type(self.client)


class ExploreConfig:
    """Explore configuration data class"""

    def __init__(
        self,
        batch_size: int = 500,
        prediction_cache_size: int = 0,
        interpret: bool = False,
        force_delete: bool = True,
        **metadata,
    ):
        self.batch_size = batch_size
        self.prediction_cache = prediction_cache_size
        self.interpret = interpret
        self.force_delete = force_delete
        self.metadata = metadata
