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
import logging

from allennlp.commands.train import Train as AllenTrain
from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.models.model import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train(AllenTrain):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = super(Train, self).add_subparser(name, parser)
        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    train_model_from_file(args.param_path, args.serialization_dir, args.overrides)


def train_model_from_file(parameter_filename: str, serialization_dir: str, overrides: str = "",
                          vocab_path: str = None) -> Model:
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, file_friendly_logging=True)
