import argparse
import logging
import os
import tarfile
from tempfile import mkdtemp
from typing import List, Dict

from allennlp.commands import Subcommand
from gevent.pywsgi import WSGIServer

from .app import make_app

__logger = logging.getLogger(__name__)


class BiomeExplore(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = 'Explore your data with model annotations'

        subparser = parser.add_parser(name, description=description, help='Explore your data')
        subparser.set_defaults(func=_explore)

        return subparser


def select_index(es_host: str, index_prefix: str = 'prediction') -> str:
    import requests
    import inquirer

    indices: List[Dict] = requests.get('{}/_cat/indices?format=json'.format(es_host)).json()
    prediction_indexes = [item['index'] for item in
                          filter(
                              lambda item: item.get('index', '').startswith(index_prefix),
                              indices
                          )]

    answers_name = 'Predicions'
    questions = [
        inquirer.List(answers_name,
                      message="Select your predction result to explore",
                      choices=prediction_indexes,
                      ),
    ]

    return inquirer.prompt(questions)[answers_name]


def _explore(_: argparse.Namespace) -> None:
    es_host = 'http://localhost:9200'

    flask_app = make_app(
        es_endpoint='{}/{}'.format(es_host, select_index(es_host)),
        statics_dir=temporal_static_path('classifier')
    )

    http_server = WSGIServer(('0.0.0.0', 9000), flask_app)

    __logger.info("Running on http://localhost:{}".format(http_server.server_port))
    http_server.serve_forever()


def temporal_static_path(explore_view: str):
    statics_tmp = mkdtemp()

    tar_file = tarfile.open(os.path.join(os.path.dirname(__file__), 'ui', '{}.tar.gz'.format(explore_view)), "r:gz")
    tar_file.extractall(path=statics_tmp)
    tar_file.close()

    print(statics_tmp)
    return statics_tmp
