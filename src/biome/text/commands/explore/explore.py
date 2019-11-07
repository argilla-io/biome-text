import logging
import os

import argparse
import datetime
import re
import warnings

import dask

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from biome.data.sources import DataSource
from dask_elk.client import DaskElasticClient
import dask.dataframe as dd

# TODO centralize configuration
from elasticsearch import Elasticsearch

from biome.text.environment import ES_HOST, BIOME_EXPLORE_ENDPOINT
from biome.text.pipelines.pipeline import Pipeline
from biome.text.interpreters import IntegratedGradient

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

BIOME_METADATA_INDEX = ".biome"

# This is the biome explore UI endpoint, used for show information
# about explorations once the data is persisted
EXPLORE_APP_ENDPOINT = os.getenv(BIOME_EXPLORE_ENDPOINT, "http://localhost:8080")

DEFAULT_INTERPRETER_CLS = IntegratedGradient


class BiomeExplore(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = """Apply a batch predictions over a dataset and make accessible through the explore UI"""

        subparser = parser.add_parser(
            name, description=description, help="Allow data exploration with prediction"
        )

        subparser.add_argument(
            "--binary",
            type=str,
            help="the archived model to make predictions with",
            required=True,
        )

        subparser.add_argument(
            "--from-source",
            type=str,
            help="datasource source definition",
            required=True,
        )

        subparser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="The batch size to use for processing",
        )

        subparser.add_argument(
            "--interpret",
            action="store_true",
            help="Add interpretation information to classifier predictions",
        )

        subparser.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.set_defaults(func=explore_with_args)

        return subparser


def explore_with_args(args: argparse.Namespace) -> None:
    def sanizite_index(index_name: str) -> str:
        return re.sub(r"\W", "_", index_name)

    model_name = os.path.basename(args.from_source)
    ds_name = os.path.dirname(os.path.abspath(args.binary))
    index = sanizite_index("{}_{}".format(model_name, ds_name).lower())

    explore(
        args.binary,
        source_path=args.from_source,
        # TODO use the /elastic explorer UI proxy as default elasticsearch endpoint
        es_host= os.getenv(ES_HOST, "http://localhost:9200"),
        es_index=index,
        interpret=args.interpret
    )


def explore(
    binary: str, 
    source_path: str, 
    es_host: str, 
    es_index: str, 
    batch_size: int = 500, 
    interpret: bool = False
) -> None:
    """
    Read a data source and tries to apply a model predictions to the whole data source. The
    results will be persisted into an elasticsearch index for further data exploration

    Parameters
    ----------
    binary
        The model.tar.gz file
    source_path
        The input data source
    es_host
        The elasticsearch host where publish the data
    es_index
        The elasticsearch index where publish the data
    batch_size
        The batch size for model predictions

    """

    pipeline = Pipeline.load(binary)

    if not isinstance(pipeline, Pipeline):
        raise ConfigurationError(
            f"Cannot load a biome Pipeline from {binary}"
            "\nPlease, be sure your pipeline class is registered as an allennlp.predictos.Predictor"
            "\nwith the same name that your model."
        )

    client = Elasticsearch(hosts=es_host, retry_on_timeout=True, http_compress=True)
    doc_type = get_compatible_doc_type(client)

    ds = DataSource.from_yaml(source_path)
    ddf_mapped = ds.to_mapped_dataframe()
    # this only makes really sense when we have a predict_batch_json method implemented ...
    npartitions = max(1, round(len(ddf_mapped) / batch_size))
    
    # a persist is necessary here, otherwise it fails for npartitions == 1
    # the reason is that with only 1 partition we pass on a generator to predict_batch_json
    ddf_mapped = ddf_mapped.repartition(npartitions=npartitions).persist()

 
    ddf_mapped["annotation"] = ddf_mapped.apply(
        lambda x: pipeline.predict_json(x.to_dict()), axis=1, meta=(None, object)
    )
    
    # TODO use map_partitions
    if interpret:
        ddf_mapped["interpretations"] = ddf_mapped.apply(
            _interpret, args=[pipeline], axis=1, meta=(None, object)
        )

    ddf_source = ds.to_dataframe()
    ddf_source = ddf_source.repartition(npartitions=npartitions).persist()

    # We are sure that both data frames are aligned!
    # A 100% safe way would be to set_index of both data frames on a meaningful column.
    # The main problem are multiple csv files (read_csv("*.csv")), where the index starts from 0 for each file ...
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ddf = dd.concat([ddf_source, ddf_mapped], axis=1)

    ddf = DaskElasticClient(
        host=es_host, retry_on_timeout=True, http_compress=True
    ).save(ddf, index=es_index, doc_type=doc_type)

    register_biome_prediction(
        name=es_index,
        es_hosts=es_host,
        created_index=es_index,
        columns=ddf.columns.values.tolist(),
        kind="explore",
        # extra metadata must be normalized
        pipeline=pipeline,
    )

    __prepare_es_index(client, es_index, doc_type)
    ddf.persist()
    # TODO: The explore APP endpoint returns localhost:8080, running biome ui defaults to
    _logger.info(
        "Data annotated successfully. You can explore your data here:"
        f"{EXPLORE_APP_ENDPOINT}/explore/{es_index}"
    )

def _interpret(x, pipeline):
    x = x.to_dict()    
    interpreter = DEFAULT_INTERPRETER_CLS(pipeline)
    # TODO This is not needed if we use 2 separate dfs
    x.pop("annotation")
    # TODO return dict when len(list) == 0
    return interpreter.saliency_interpret_from_json(x)

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


def register_biome_prediction(
    name: str, pipeline: Pipeline, es_hosts: str, created_index: str, **kwargs
):
    """
    Creates a new metadata entry for the incoming prediction

    Parameters
    ----------
    name
        A descriptive prediction name
    pipeline
        The pipeline used for the prediction batch
    created_index
        The elasticsearch index created for the prediction
    es_hosts
        The elasticsearch host where publish the new entry
    kwargs
        Extra arguments passed as extra metadata info
    """

    metadata_index = f"{BIOME_METADATA_INDEX}"
    es_client = Elasticsearch(hosts=es_hosts, retry_on_timeout=True, http_compress=True)

    es_client.indices.create(
        index=metadata_index,
        body={"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}},
        ignore=400,
    )

    predict_signature = [
        k for k, v in pipeline.signature.items() if not v.get("optional")
    ]
    parameters = {
        **kwargs,
        "pipeline": pipeline.name,
        "signature": [k for k in pipeline.signature.keys()],
        "predict_signature": predict_signature,
        # TODO remove when ui is adapted
        "inputs": predict_signature,  # backward compatibility
    }

    es_client.update(
        index=metadata_index,
        doc_type=get_compatible_doc_type(es_client),
        id=created_index,
        body={
            "doc": dict(name=name, created_at=datetime.datetime.now(), **parameters),
            "doc_as_upsert": True,
        },
    )
    del es_client


def __prepare_es_index(es_client: Elasticsearch, index: str, doc_type: str):
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

    es_client.indices.delete(index=index, ignore=[400, 404])
    es_client.indices.create(
        index=index,
        body={"mappings": {doc_type: {"dynamic_templates": dynamic_templates}}},
        ignore=400,
    )
