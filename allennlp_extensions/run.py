import argparse
import logging
import os
import sys
from pydoc import locate
from typing import Iterable

import fire
from allennlp.commands import Train, Evaluate

from allennlp_extensions.commands.restapi import RestAPI
from allennlp_extensions.commands.kafka import KafkaPipelineCommand
from allennlp_extensions.commands.publish import PublishModel

__logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))


def configure_logging(enable_debug: bool = False):
    level = logging.DEBUG if enable_debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=level)


def load_customs_components_from_file(subcommand: str):
    def load_customs(names: Iterable[str]):
        for name in names:
            locate(name)
            __logger.info('Loaded custom class %s' % name)

    def load_from_file(filename: str):
        with open(filename) as customs_definitions:
            load_customs([definition for definition in customs_definitions])

    try:
        load_from_file('custom.%s.ini' % subcommand)
    except:
        try:
            load_from_file('custom.ini')
        except:
            '''Nothing to do'''


def main(*kwargs) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes
    in the ``allennlp`` codebase. In particular, once you start creating your own
    ``Model`` s and so forth, it won't work for them. However, ``allennlp.run`` is
    simply a wrapper around this function. To use the command line interface with your
    own custom classes, just create your own script that imports all of the classes you want
    and then calls ``main()``.

    The default models for ``serve`` and the default predictors for ``predict`` are
    defined above. If you'd like to add more or use different ones, the
    ``model_overrides`` and ``predictor_overrides`` arguments will take precedence over the defaults.
    """
    # pylint: disable=dangerous-default-value

    parser = argparse.ArgumentParser(description="Run AllenNLP", usage='%(prog)s [command]', prog=__name__)
    parser.add_argument('-v', '--verbose', action='store_true', dest='enable_debug', default=False, help="Enables debug traces")
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    from allennlp_extensions.commands.describe import DescribeRegistrable
    subcommands = {
        # Default commands
        "train": Train(),
        'publish': PublishModel(),
        'evaluate': Evaluate(),
        "rest": RestAPI(),
        'kafka': KafkaPipelineCommand(),
        'describe': DescribeRegistrable()
    }

    for name, subcommand in subcommands.items():
        subcommand.add_subparser(name, subparsers)

    args = parser.parse_args()
    configure_logging(args.enable_debug)

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        load_customs_components_from_file(kwargs[0])
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    fire.Fire(main)
