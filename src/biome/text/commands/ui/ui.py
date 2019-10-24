import argparse
import logging
import os
import tarfile
from tempfile import mkdtemp
from typing import Optional

from allennlp.commands import Subcommand
from gevent.pywsgi import WSGIServer

from biome.data.utils import ENV_ES_HOSTS
from .app import make_app

# TODO centralize configuration
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class BiomeUI(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = "Explore your data with model annotations"

        subparser = parser.add_parser(
            name, description=description, help="Explore your data"
        )
        subparser.add_argument(
            "--es-host", help="The elasticsearch backend storage", default=None
        )
        subparser.add_argument(
            "--port",
            help="Listening port for application",
            default=9000,
            type=lambda a: int(a),
        )
        subparser.set_defaults(func=launch_ui_from_args)
        return subparser


def launch_ui_from_args(args: argparse.Namespace) -> None:
    return launch_ui(args.es_host, port=args.port)


def launch_ui(es_host: str, port: int = 9000) -> None:
    es_host = es_host if es_host else os.getenv(ENV_ES_HOSTS, "http://localhost:9200")

    flask_app = make_app(
        es_host=es_host, statics_dir=temporal_static_path("classifier")
    )

    http_server = WSGIServer(("0.0.0.0", port), flask_app)
    _logger.info(
        f"Running biome UI on http://0.0.0.0:{http_server.server_port} with elasticsearch backend {es_host}"
    )
    http_server.serve_forever()


def temporal_static_path(explore_view: str, basedir: Optional[str] = None):
    statics_tmp = mkdtemp()

    compressed_ui = os.path.join(
        basedir or os.path.dirname(__file__), "{}.tar.gz".format(explore_view)
    )
    tar_file = tarfile.open(compressed_ui, "r:gz")
    tar_file.extractall(path=statics_tmp)
    tar_file.close()

    return statics_tmp
