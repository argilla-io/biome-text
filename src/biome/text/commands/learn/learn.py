"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp.run train --help
   usage: run [command] train [-h] -s SERIALIZATION_DIR param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be trained

   optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the model and its logs
"""

import argparse
from typing import Callable

from allennlp.commands import Subcommand

from biome.text import Pipeline
from biome.text.pipelines._impl.allennlp.learn.default_callback_trainer import (
    DefaultCallbackTrainer,
)

__alias__ = [DefaultCallbackTrainer]


class BiomeLearn(Subcommand):
    @staticmethod
    def description() -> str:
        return "Make a model learn"

    @staticmethod
    def command_handler() -> Callable:
        return learn_from_args

    def add_subparser(
        self,
        name: str,
        parser: argparse._SubParsersAction,  # pylint: disable=protected-access
    ) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
            name, description=self.description(), help=self.description()
        )

        subparser.add_argument(
            "--spec",
            type=str,
            help="model.yml specification",
            required=False,
            default=None,
        )
        subparser.add_argument(
            "--binary",
            type=str,
            help="pretrained model binary tar.gz",
            required=False,
            default=None,
        )

        subparser.add_argument(
            "--vocab",
            type=str,
            help="path to existing vocab",
            required=False,
            default=None,
        )

        subparser.add_argument(
            "--trainer", type=str, help="trainer.yml specification", required=True
        )
        subparser.add_argument(
            "--train", type=str, help="train datasource definition", required=True
        )
        subparser.add_argument(
            "--validation",
            type=str,
            help="validation datasource source definition",
            required=False,
            default=None,
        )
        subparser.add_argument(
            "--test",
            type=str,
            help="test datasource source definition",
            required=False,
            default=None,
        )

        subparser.add_argument(
            "--output",
            type=str,
            help="learn process generation folder",
            required=True,
            default=None,
        )

        subparser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Turn on verbose logs from AllenNLP.",
        )

        subparser.set_defaults(func=self.command_handler())

        return subparser


def learn_from_args(args: argparse.Namespace):
    """Launches a pipeline learn action with input command line arguments"""

    if args.spec:
        pipeline = Pipeline.from_config(args.spec)

    elif args.binary:
        pipeline = Pipeline.load(args.binary)
    else:
        raise ValueError("Missing parameter --spec/--binary")

    return pipeline.learn(
        output=args.output,
        vocab=args.vocab,
        trainer=args.trainer,
        train=args.train,
        validation=args.validation,
        test=args.test,
        verbose=args.verbose,
    )
