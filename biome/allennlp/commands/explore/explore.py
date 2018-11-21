import argparse
import datetime
import logging
from multiprocessing import Process

from . import web_server
from ..predict.predict import BiomePredict, _predict

__logger = logging.getLogger(__name__)


class BiomeExplore(BiomePredict):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = 'Explore your data with model annotations'

        subparser = parser.add_parser(name, description=description, help='Explore your data')
        self.configure_parser(subparser)
        subparser.set_defaults(func=_explore)

        return subparser


def _explore(args: argparse.Namespace) -> None:
    now = datetime.datetime.now()
    args.to_sink = dict(index='explore_{}'.format(now.strftime('%Y%m%d%H%M%S%f')),
                        type='docs',
                        es_hosts='localhost:9200',
                        es_batch_size=args.batch_size)

    web_server_process = Process(target=web_server.start)
    web_server_process.start()
    _predict(args)
    web_server_process.join()
