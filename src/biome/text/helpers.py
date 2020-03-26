from typing import Dict, Any

import yaml
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


def yaml_to_dict(file: str) -> Dict[str, Any]:
    """
    Loads a yaml file as python dict

    Parameters
    ----------

    file:str
        The yaml file path

    Returns
    -------
        A dictionary with yaml data

    """
    with open(file) as yaml_content:
        config = yaml.safe_load(yaml_content)

    return config
