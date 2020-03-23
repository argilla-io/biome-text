from elasticsearch import Elasticsearch


def get_compatible_doc_type(client: Elasticsearch) -> str:
    """
    Find a compatible name for doc type by checking the cluster info
    Parameters
    ----------
    client
        The elasticsearch client

    Returns
    -------
        A compatible name for doc type in function of cluster version
    """

    es_version = int(client.info()["version"]["number"].split(".")[0])
    return "_doc" if es_version >= 6 else "doc"
