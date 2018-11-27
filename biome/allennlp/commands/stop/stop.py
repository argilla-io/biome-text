import argparse
import logging

from allennlp.commands import Subcommand
from biome.allennlp.commands.start.start import ES_VERSION
from elasticsearch_runner.runner import ElasticsearchRunner

__logger = logging.getLogger(__name__)


class BiomeStop(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = 'Stop all resources needed for biome your model'

        subparser = parser.add_parser(name, description=description, help=description)
        subparser.set_defaults(func=init)

        return subparser


def init(args: argparse.Namespace) -> None:
    es_runner = ElasticsearchRunner(version=ES_VERSION)
    es_runner.run()
    es_runner.stop()
