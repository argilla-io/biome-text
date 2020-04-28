import copy
import datetime
import inspect
import os
from inspect import Parameter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import yaml
from allennlp.common import Params, Registrable
from allennlp.common.from_params import remove_optional
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, _Seq2VecWrapper
from elasticsearch import Elasticsearch

from . import constants, environment


def yaml_to_dict(filepath: str) -> Dict[str, Any]:
    """Loads a yaml file into a data dictionary"""
    with open(filepath) as yaml_content:
        config = yaml.safe_load(yaml_content)
    return config


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


def get_env_cuda_device() -> int:
    """Gets the cuda device from an environment variable.

    This is necessary to activate a GPU if available

    Returns
    -------
    cuda_device
        The integer number of the CUDA device
    """
    cuda_device = int(os.getenv(environment.CUDA_DEVICE, "-1"))
    return cuda_device


class ElasticsearchExplore:
    """Elasticsearch data exploration class"""

    def __init__(self, es_index: str, es_host: Optional[str] = None):
        self.es_index = es_index
        self.es_host = es_host or constants.DEFAULT_ES_HOST
        if not self.es_host.startswith("http"):
            self.es_host = f"http://{self.es_host}"

        self.client = Elasticsearch(
            hosts=es_host, retry_on_timeout=True, http_compress=True
        )
        self.es_doc = get_compatible_doc_type(self.client)

    def create_explore_data_record(self, parameters: Dict[str, Any]):
        """Creates an exploration data record data exploration"""

        self.client.indices.create(
            index=constants.BIOME_METADATA_INDEX,
            body={
                "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}
            },
            params=dict(ignore=400),
        )

        self.client.update(
            index=constants.BIOME_METADATA_INDEX,
            doc_type=constants.BIOME_METADATA_INDEX_DOC,
            id=self.es_index,
            body={
                "doc": dict(
                    name=self.es_index, created_at=datetime.datetime.now(), **parameters
                ),
                "doc_as_upsert": True,
            },
        )

    def create_explore_data_index(self, force_delete: bool):
        """Creates an explore data index if not exists or is forced"""
        dynamic_templates = [
            {
                data_type: {
                    "match_mapping_type": data_type,
                    "path_match": path_match,
                    "mapping": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                }
            }
            for data_type, path_match in [("*", "*.value"), ("string", "*")]
        ]

        if force_delete:
            self.client.indices.delete(index=self.es_index, ignore=[400, 404])

        self.client.indices.create(
            index=self.es_index,
            body={"mappings": {self.es_doc: {"dynamic_templates": dynamic_templates}}},
            ignore=400,
        )


def update_method_signature(signature: inspect.Signature, to_method):
    """Updates signature to method"""

    def wrapper(*args, **kwargs):
        return to_method(*args, **kwargs)

    wrapper.__signature__ = signature
    return wrapper


def isgeneric(class_type: Type) -> bool:
    """Checks if a class type is a generic type (List[str] or Union[str, int]"""
    return hasattr(class_type, "__origin__")


def is_running_on_notebook() -> bool:
    """Checks if code is running inside a jupyter notebook"""
    try:
        import IPython

        return IPython.get_ipython().has_trait("kernel")
    except (AttributeError, NameError, ModuleNotFoundError):
        return False


def split_signature_params_by_predicate(
    signature_function: Callable, predicate: Callable
) -> Tuple[List[Parameter], List[Parameter]]:
    """Splits parameters signature by defined boolean predicate function"""
    signature = inspect.signature(signature_function)
    parameters = list(
        filter(
            lambda p: p.name != "self"
            and p.kind not in [Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL],
            signature.parameters.values(),
        )
    )
    matches_group = list(filter(lambda p: predicate(p), parameters))
    non_matches_group = list(filter(lambda p: not predicate(p), parameters))

    return matches_group, non_matches_group
