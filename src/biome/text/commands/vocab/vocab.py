"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp_2.run train --help
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
from copy import deepcopy
from typing import Optional, Callable

from allennlp.commands.make_vocab import make_vocab_from_params
from allennlp.common.params import Params

from biome.text.commands.helpers import BiomeConfig
from biome.text.commands import BiomeLearn
from biome.data.utils import configure_dask_cluster

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BiomeVocab(BiomeLearn):
    def description(self) -> str:
        return "Build a vocabulary"

    def command_handler(self) -> Callable:
        return vocab_from_args


def vocab_from_args(args: argparse.Namespace):
    vocab(
        spec=args.spec,
        train=args.train,
        validation=args.validation,
        test=args.test,
        output=args.output,
    )


def vocab(
    spec: Optional[str], train: str, validation: str, test: Optional[str], output: str
):
    allennlp_configuration = BiomeConfig(
        model_path=spec,
        trainer_path=None,
        train_path=train,
        validation_path=validation,
        test_path=test,
    ).to_allennlp_params()

    configure_dask_cluster()
    make_vocab_from_params(Params(deepcopy(allennlp_configuration)), output)
