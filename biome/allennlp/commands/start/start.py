import argparse
import logging

from elasticsearch_runner.runner import ElasticsearchRunner

from allennlp.commands import Subcommand

__logger = logging.getLogger(__name__)

ES_VERSION = '6.3.2'


class BiomeStart(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = 'Initialize all resources needed for biome your model'

        subparser = parser.add_parser(name, description=description, help=description)
        subparser.set_defaults(func=init)

        return subparser


def init(args: argparse.Namespace) -> None:
    es_runner = ElasticsearchRunner(version=ES_VERSION)
    es_runner.install()
    es_runner.run()
