import argparse
import logging

from allennlp.commands import Subcommand
from allennlp.models import load_archive
from allennlp.service import server_simple
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from biome.allennlp.predictors.utils import get_predictor_from_archive

logger = logging.getLogger(__name__)


class BiomeRestAPI(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the web service, which provides an HTTP Rest API.'''
        subparser = parser.add_parser(name, description=description, help='Run the web service.')

        subparser.add_argument('--port', type=int, default=8000)
        subparser.add_argument('--binary', type=str, required=True)

        subparser.set_defaults(func=_serve)

        return subparser


def _serve(args: argparse.Namespace) -> None:
    model_archive = load_archive(args.binary)
    # Matching predictor name with model name
    predictor = get_predictor_from_archive(model_archive)

    # server_sanic.run(args.port, args.workers, model_archive=args.binary)
    app = server_simple.make_app(predictor, title=predictor.__class__.__name__)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    logger.info(f"Model loaded, serving demo on port {args.port}")
    http_server.serve_forever()
