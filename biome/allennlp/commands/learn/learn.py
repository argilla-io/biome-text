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

import yaml
from allennlp.commands import Subcommand
from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from typing import Optional, Dict, Any

from biome.data.utils import read_definition_from_model_spec, configure_dask_cluster

__logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MODEL_FIELD = 'model'
TRAIN_DATA_FIELD = 'train_data_path'
VALIDATION_DATA_FIELD = 'validation_data_path'
TEST_DATA_FIELD = 'test_data_path'


class BiomeLearn(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(name, description='Make a model learn', help='Make a model learn')

        subparser.add_argument('--spec', type=str, help='model.yml specification', required=False)
        subparser.add_argument('--binary', type=str, help='pretrained model binary tar.gz', required=False)

        subparser.add_argument('--trainer', type=str, help='trainer.yml specification', required=True)
        subparser.add_argument('--train', type=str, help='train datasource definition', required=True)
        subparser.add_argument('--validation', type=str, help='validation datasource source definition', required=True)
        subparser.add_argument('--test', type=str, help='test datasource source definition', required=False)

        subparser.add_argument('--output', type=str, help='learn process generation folder', required=True)

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    train_model_from_file(
        model_spec=args.spec,
        model_binary=args.binary,
        trainer_path=args.trainer,
        train_cfg=args.train,
        validation_cfg=args.validation,
        test_cfg=args.test,
        output=args.output
    )


def train_model_from_file(output: str,
                          model_spec: Optional[str] = None,
                          model_binary: Optional[str] = None,
                          trainer_path: str = '',
                          train_cfg: str = '',
                          validation_cfg: str = '',
                          test_cfg: Optional[str] = None) -> Model:
    if not model_binary and not model_spec:
        raise Exception('Missing parameter --spec/--binary')

    with open(trainer_path) as trainer_file:
        trainer_params = yaml.load(trainer_file)
        cfg_params = __load_from_archive(model_binary) \
            if model_binary \
            else read_definition_from_model_spec(model_spec) if model_spec else dict()

    allennlp_configuration = {
        **cfg_params,
        **trainer_params,
        TRAIN_DATA_FIELD: train_cfg,
        VALIDATION_DATA_FIELD: validation_cfg
    }

    if test_cfg and not test_cfg.isspace():
        allennlp_configuration.update({TEST_DATA_FIELD: test_cfg})

    allennlp_configuration[MODEL_FIELD] = __merge_model_params(model_binary, allennlp_configuration.get(MODEL_FIELD))
    params = Params(allennlp_configuration)

    __logger.info('Launching dask cluster')
    configure_dask_cluster()

    return train_model(params, output, file_friendly_logging=True, recover=False, force=True)


def __load_from_archive(model_binary: str) -> Dict[str, Any]:
    archive = load_archive(model_binary)
    return archive.config.as_dict()


def __merge_model_params(model_location: Optional[str], model_params: Dict[str, Any]) -> Dict:
    return dict(**model_params, model_location=model_location) \
        if model_location \
        else model_params
