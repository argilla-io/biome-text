import argparse
import logging

from allennlp.commands import Subcommand

from biome.helpers import create_es_runner

__logger = logging.getLogger(__name__)


class BiomeStop(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = 'Stop all resources needed for biome your model'

        subparser = parser.add_parser(name, description=description, help=description)
        subparser.set_defaults(func=init)

        return subparser


def init(args: argparse.Namespace) -> None:
    es_runner = create_es_runner()
    es_runner.stop()
