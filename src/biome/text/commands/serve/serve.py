import argparse
import logging

from allennlp.commands import Subcommand

from biome.text import Pipeline

__LOGGER = logging.getLogger(__name__)


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
        subparser.add_argument(
            "--output",
            type=str,
            default=None,
            required=False,
            help="If provided, write predictions as json file to this output folder.",
        )
        subparser.set_defaults(func=_serve_from_args)

        return subparser


def _serve_from_args(args: argparse.Namespace) -> None:
    return serve(binary=args.binary, port=args.port, output=args.output)


def serve(binary: str, port: int = 8000, output: str = None) -> None:
    pipeline = Pipeline.load(binary)
    pipeline.serve(port, predictions=output)


def make_app(binary: str, output: str = None):
    """
    This function allows to serve a model from a gunicorn server. For example:

    >>>gunicorn 'biome.allennlp.commands.serve.serve:make_app("/path/to/model.tar.gz")'

    Parameters
    ----------
    binary
        Path to the *model.tar.gz* file
    output
        Path to the output folder, in which to store the predictions.

    Returns
    -------
    app
        A Flask app used by gunicorn server
    """
    pass
