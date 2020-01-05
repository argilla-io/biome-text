import argparse
import logging

from allennlp.commands import Subcommand
from allennlp.service import server_simple
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from biome.text import Pipeline

logger = logging.getLogger(__name__)


class BiomeRestAPI(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = """Run the web service, which provides an HTTP Rest API."""
        subparser = parser.add_parser(
            name, description=description, help="Run the web service."
        )

        subparser.add_argument("--port", type=int, default=8000)
        subparser.add_argument("--binary", type=str, required=True)
        subparser.set_defaults(func=_serve_from_args)

        return subparser


def _serve_from_args(args: argparse.Namespace) -> None:
    return serve(binary=args.binary, port=args.port)


def make_app(binary: str):
    """
    This function allows serve model from gunicorn server. For example:

    >>>gunicorn 'biome.allennlp.commands.serve.serve:make_app("/path/to/model.tar.gz")'

    :param binary: the model.tar.gz path
    :return: a Flask app used by gunicorn server
    """
    pipeline = Pipeline.load(binary)
    app = server_simple.make_app(pipeline, title=pipeline.__class__.__name__)
    CORS(app)
    return app


def serve(binary: str, port: int = 8000) -> None:
    app = make_app(binary)

    http_server = WSGIServer(("0.0.0.0", port), app)
    logger.info(f"Model loaded, serving on port {port}")
    http_server.serve_forever()
