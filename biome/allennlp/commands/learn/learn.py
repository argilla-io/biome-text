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
from typing import Optional, Callable

from allennlp.commands import Subcommand
from allennlp.commands.fine_tune import fine_tune_model
from allennlp.commands.train import train_model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data import DataIterator, DatasetReader
from allennlp.models.model import Model

from biome.allennlp.commands.helpers import biome2allennlp_params
from biome.allennlp.models.archival import load_archive
from biome.data.utils import configure_dask_cluster

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DATASET_READER_FIELD_NAME = 'dataset_reader'


class BiomeLearn(Subcommand):

    def description(self) -> str:
        return 'Make a model learn'

    def command_handler(self) -> Callable:
        return learn_from_args

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(name, description=self.description(), help=self.description())

        subparser.add_argument('--spec', type=str, help='model.yml specification', required=False)
        subparser.add_argument('--binary', type=str, help='pretrained model binary tar.gz', required=False)

        subparser.add_argument('--vocab', type=str, help='path to existing vocab', required=False)

        subparser.add_argument('--trainer', type=str, help='trainer.yml specification', required=True)
        subparser.add_argument('--train', type=str, help='train datasource definition', required=True)
        subparser.add_argument('--validation', type=str, help='validation datasource source definition', required=False)
        subparser.add_argument('--test', type=str, help='test datasource source definition', required=False)

        subparser.add_argument('--output', type=str, help='learn process generation folder', required=True)

        subparser.set_defaults(func=self.command_handler())

        return subparser


def learn_from_args(args: argparse.Namespace):
    learn(
        model_spec=args.spec,
        model_binary=args.binary,
        vocab=args.vocab,
        trainer_path=args.trainer,
        train_cfg=args.train,
        validation_cfg=args.validation,
        test_cfg=args.test,
        output=args.output
    )


def check_configuration(params: Params):
    DatasetReader.from_params(params.get(DATASET_READER_FIELD_NAME))
    DataIterator.from_params(params.get('iterator'))


def check_model_configuration(params: Params):
    pass
    # berModel.from_params(params.get('model'), vocab=vocab)


def learn(output: str,
          model_spec: Optional[str] = None,
          model_binary: Optional[str] = None,
          vocab: Optional[str] = None,
          trainer_path: str = '',
          train_cfg: str = '',
          validation_cfg: str = '',
          test_cfg: Optional[str] = None) -> Model:
    allennlp_configuration = biome2allennlp_params(model_spec,
                                                   trainer_path,
                                                   vocab,
                                                   train_cfg, validation_cfg, test_cfg)

    if not model_binary and not model_spec:
        raise ConfigurationError('Missing parameter --spec/--binary')

    __logger.info('Checking initial configuration')
    check_configuration(Params(deepcopy(allennlp_configuration)))

    __logger.info('Launching dask cluster')
    configure_dask_cluster(n_workers=1)

    # Vocabulary is needed for components instantiation
    __logger.info('Checking model configuration')
    check_model_configuration(Params(deepcopy(allennlp_configuration)))

    allennlp_configuration = {**allennlp_configuration}
    if model_binary:
        archive = load_archive(model_binary)
        __logger.info(archive.config.as_dict())

        return fine_tune_model(
            model=archive.model,
            params=Params({
                DATASET_READER_FIELD_NAME: archive.config.get(DATASET_READER_FIELD_NAME).as_dict(),
                **allennlp_configuration
            }),
            serialization_dir=output,
            extend_vocab=False,
            file_friendly_logging=True
        )
    else:
        return train_model(
            params=Params(allennlp_configuration),
            serialization_dir=output,
            file_friendly_logging=True,
            recover=False,
            force=True
        )
