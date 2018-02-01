import argparse

from allennlp.commands import Subcommand

from allennlp_extensions.commands.restapi import server_sanic


class RestAPI(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the web service, which provides an HTTP Rest API.'''
        subparser = parser.add_parser(name, description=description, help='Run the web service.')

        subparser.add_argument('--port', type=int, default=8000)
        subparser.add_argument('--workers', type=int, default=1)
        subparser.add_argument('--model', type=str, required=True)
        subparser.add_argument('--model-location', type=str, required=True)

        subparser.set_defaults(func=_serve)

        return subparser


def _serve(args: argparse.Namespace) -> None:
    server_sanic.run(args.port, args.workers, trained_models={
        args.model: args.model_location
    })
