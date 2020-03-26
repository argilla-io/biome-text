import argparse
import logging
import os
import re

from allennlp.commands.subcommand import Subcommand
from biome.data import DataSource

from biome.text.constants import BIOME_METADATA_INDEX, EXPLORE_APP_ENDPOINT
from biome.text.environment import ES_HOST
from biome.text.helpers import get_compatible_doc_type
from biome.text.pipelines import explore as pipeline_explore
from biome.text.pipelines.explore import ElasticsearchConfig, ExploreConfig
from biome.text import Pipeline

# TODO centralize configuration

# TODO: Remove (Just backward compatibility)
__alias__ = [get_compatible_doc_type, BIOME_METADATA_INDEX, EXPLORE_APP_ENDPOINT]


logging.basicConfig(level=logging.INFO)
__LOGGER = logging.getLogger(__name__)


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
            "--prediction-cache-size",
            type=int,
            default=0,
            help="Size of the prediction cache in number of items",
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
        es_host=os.getenv(ES_HOST, "http://localhost:9200"),
        es_index=index,
        prediction_cache_size=args.prediction_cache_size,
        interpret=args.interpret,
    )


# TODO: Remove (Just backward compatibility)
def explore(
    binary: str,
    source_path: str,
    es_host: str,
    es_index: str,
    batch_size: int = 500,
    prediction_cache_size: int = 0,
    interpret: bool = False,
    force_delete: bool = True,
    **prediction_metadata,
) -> None:
    pipeline = Pipeline.load(binary)
    pipeline.explore(
        ds_path=source_path,
        # TODO use the /elastic explorer UI proxy as default elasticsearch endpoint
        es_config=ElasticsearchConfig(es_host=es_host, es_index=es_index),
        config=ExploreConfig(
            batch_size=batch_size,
            prediction_cache_size=prediction_cache_size,
            interpret=interpret,
            force_delete=force_delete,
            **prediction_metadata,
        ),
    )


# TODO: Remove (Just backward compatibility)
def register_biome_prediction(
    name: str, created_index: str, es_hosts: str, pipeline: Pipeline, **extra_args: dict
) -> None:
    pipeline_explore.register_biome_prediction(
        name, pipeline, ElasticsearchConfig(es_hosts, created_index), **extra_args
    )
