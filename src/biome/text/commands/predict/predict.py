import argparse
import datetime
import logging
import os
import re

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from biome.data.sources import DataSource
from biome.data.utils import ENV_ES_HOSTS
from dask_elk.client import DaskElasticClient

# TODO centralize configuration
from elasticsearch import Elasticsearch

from biome.text.pipelines.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

BIOME_METADATA_INDEX = ".biome"


class BiomePredict(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = """Make a batch prediction over input test data set"""

        subparser = parser.add_parser(
            name,
            description=description,
            help="Use a trained model to make predictions.",
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
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.set_defaults(func=_predict)

        return subparser


def _predict(args: argparse.Namespace) -> None:
    def sanizite_index(index_name: str) -> str:
        return re.sub(r"\W", "_", index_name)

    ds_name = os.path.basename(args.binary)
    model_name = os.path.dirname(args.from_source)
    index = sanizite_index("{}_{}".format(model_name, ds_name).lower())

    predict(
        args.binary,
        source_path=args.from_source,
        es_host=os.getenv(ENV_ES_HOSTS, "http://localhost:9200"),
        es_index=index,
    )


def predict(
    binary: str,
    source_path: str,
    es_host: str,
    es_index: str,
    es_doc: str = "_doc",
    batch_size: int = 500,
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
    es_doc
        The mapping type where publish the data
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

    es_client = DaskElasticClient(
        host=es_host, retry_on_timeout=True, http_compress=True
    )

    ds = DataSource.from_yaml(source_path)
    ddf = ds.to_forward_dataframe()
    npartitions = max(1, round(len(ddf) / batch_size))
    # a persist is necessary here, otherwise it fails for npartitions == 1
    # the reason is that with only 1 partition we pass on a generator to predict_batch_json
    ddf = ddf.repartition(npartitions=npartitions).persist()
    ddf["annotation"] = ddf.apply(pipeline.predict_json, axis=1, meta=object)
    ddf = es_client.save(ddf, index=es_index, doc_type=es_doc)

    register_biome_prediction(
        name=es_index,
        created_index=es_index,
        es_hosts=es_host,
        # extra metadata must be normalized
        pipeline=pipeline.name,
        signature=pipeline.reader.signature,
        # TODO remove when ui is adapted
        inputs=pipeline.reader.signature,  # backward compatibility
        columns=ddf.columns.values.tolist(),
        kind="explore",
    )

    __prepare_es_index(es_host, es_index, es_doc)
    ddf.persist()


def register_biome_prediction(name: str, es_hosts: str, created_index: str, **kwargs):
    """
    Creates a new metadata entry for the incoming prediction

    Parameters
    ----------
    name
        A descriptive prediction name
    created_index
        The elasticsearch index created for the prediction
    es_hosts
        The elasticsearch host where publish the new entry
    kwargs
        Extra arguments passed as extra metadata info

    """

    metadata_index = f"{BIOME_METADATA_INDEX}"
    es_client = Elasticsearch(hosts=es_hosts, retry_on_timeout=True, http_compress=True)
    es_version = int(es_client.info()["version"]["number"].split(".")[0])

    es_client.indices.create(
        index=metadata_index,
        body={"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}},
        ignore=400,
    )

    es_client.update(
        index=metadata_index,
        doc_type="_doc" if es_version >= 6 else "doc",
        id=created_index,
        body={
            "doc": dict(name=name, created_at=datetime.datetime.now(), **kwargs),
            "doc_as_upsert": True,
        },
    )
    del es_client


def __prepare_es_index(es_hosts: str, index: str, doc_type: str):
    es_client = Elasticsearch(hosts=es_hosts, retry_on_timeout=True, http_compress=True)

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
