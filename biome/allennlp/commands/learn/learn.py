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
from copy import deepcopy

from allennlp.commands import Subcommand
from allennlp.commands.dry_run import dry_run_from_params
from allennlp.commands.train import train_model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.models.model import Model
from typing import Optional, Callable

from biome.allennlp.commands.helpers import biome2allennlp_params
from biome.data.utils import configure_dask_cluster

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BiomeLearn(Subcommand):

    def description(self) -> str:
        return 'Make a model learn'

    def command_handler(self) -> Callable:
        return learn_from_args

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(name, description=self.description(), help=self.description())

        subparser.add_argument('--spec', type=str, help='model.yml specification', required=False)
        subparser.add_argument('--binary', type=str, help='pretrained model binary tar.gz', required=False)

        subparser.add_argument('--trainer', type=str, help='trainer.yml specification', required=True)
        subparser.add_argument('--train', type=str, help='train datasource definition', required=True)
        subparser.add_argument('--validation', type=str, help='validation datasource source definition', required=True)
        subparser.add_argument('--test', type=str, help='test datasource source definition', required=False)

        subparser.add_argument('--output', type=str, help='learn process generation folder', required=True)

        subparser.set_defaults(func=self.command_handler())

        return subparser


def learn_from_args(args: argparse.Namespace):
    learn(
        model_spec=args.spec,
        model_binary=args.binary,
        trainer_path=args.trainer,
        train_cfg=args.train,
        validation_cfg=args.validation,
        test_cfg=args.test,
        output=args.output
    )


def check_configuration(params: Params):
    DatasetReader.from_params(params.get('dataset_reader'))
    DataIterator.from_params(params.get('iterator'))


def check_model_configuration(params: Params, vocab: Vocabulary):
    Model.from_params(params.get('model'), vocab=vocab)


def learn(output: str,
          model_spec: Optional[str] = None,
          model_binary: Optional[str] = None,
          trainer_path: str = '',
          train_cfg: str = '',
          validation_cfg: str = '',
          test_cfg: Optional[str] = None) -> Model:
    allennlp_configuration = biome2allennlp_params(model_spec,
                                                   model_binary,
                                                   trainer_path,
                                                   train_cfg, validation_cfg, test_cfg)

    __logger.info('Checking initial configuration')
    check_configuration(Params(deepcopy(allennlp_configuration)))

    __logger.info('Launching dask cluster')
    configure_dask_cluster()

    vocab_dir = '{}.vocab'.format(output)
    vocabulary_configuration = dict(directory_path='{}/vocabulary'.format(vocab_dir))
    try:
        dry_run_from_params(Params(deepcopy(allennlp_configuration)), vocab_dir)
    except ConfigurationError as cerr:
        if 'serialization directory is non-empty' not in cerr.message:
            raise cerr

    # Vocabulary is needed for components instantiation
    __logger.info('Checking model configuration')
    check_model_configuration(Params(deepcopy(allennlp_configuration)),
                              Vocabulary.from_params(Params(deepcopy(vocabulary_configuration))))

    allennlp_configuration = {**allennlp_configuration, 'vocabulary': vocabulary_configuration}
    return train_model(Params(allennlp_configuration), output, file_friendly_logging=True, recover=False, force=True)
