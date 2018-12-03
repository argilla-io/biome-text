from elasticsearch_runner.runner import ElasticsearchRunner

ES_VERSION = '6.3.2'


def create_es_runner() -> ElasticsearchRunner:
    es_runner = ElasticsearchRunner(version=ES_VERSION)
    es_runner.install()
    es_runner.run()

    return es_runner
