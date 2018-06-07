import argparse
import logging
import os
import sys
from numbers import Number
from pydoc import locate
from typing import Iterable

from dask.distributed import Client
from dask.cache import Cache
import dask
import fire
from allennlp.commands import Evaluate

from recognai.commands.predict.predict import Predict
from recognai.commands.publish import PublishModel
from recognai.commands.restapi import RestAPI
from recognai.commands.train import RecognaiTrain

__logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

DEFAULT_DASK_CLUSTER = '127.0.0.1:8786'
DEFAULT_DASK_CACHE_SIZE = 2e9
DEFAULT_DASK_BLOCK_SIZE = 50e6


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

    parser = argparse.ArgumentParser(description="Run RecognAI", usage='%(prog)s [command]', prog=__name__)
    parser.add_argument('--dask', dest='dask_cluster', default=DEFAULT_DASK_CLUSTER, help='Dask cluster endpoint')
    parser.add_argument('--dask-cache', dest='dask_cache_size', default=DEFAULT_DASK_CACHE_SIZE)
    parser.add_argument('--dask-block-size', dest='dask_block_size', default=DEFAULT_DASK_BLOCK_SIZE)
    parser.add_argument('-v', '--VERBOSE', action='store_true', dest='enable_debug', default=False,
                        help="Enables debug traces")
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    from recognai.commands.describe import DescribeRegistrable
    subcommands = {
        # Default commands
        # "preprocess": Preprocess(),
        # "make-vocab": MakeVocab(),
        "train": RecognaiTrain(),
        'predict': Predict(),
        'evaluate': Evaluate(),
        'publish': PublishModel(),
        'rest': RestAPI(),
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
        dask_client = None

        try:
            if __dask_needed(args.func.__name__):
                dask_client = _dask_client(args.dask_cluster, args.dask_cache_size)

            load_customs_components_from_file(kwargs[0])
            args.func(args)
        finally:
            try:
                dask_client.close()
            except:
                pass
    else:
        parser.print_help()


def __dask_needed(func_name: str) -> bool:

    for dasked_command in ['predict', 'train', 'evaluate']:
        if dasked_command in func_name:
            return True
    return False


def _dask_client(dask_cluster: str, cache_size: Number) -> Client:
    if cache_size:
        cache = Cache(cache_size)
        cache.register()

    try:
        return dask.distributed.Client(dask_cluster)
    except:
        return dask.distributed.Client()


if __name__ == '__main__':
    fire.Fire(main)
