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
import glob
import logging
import os

import argparse
import shutil
from allennlp.commands import Subcommand
from allennlp.commands.fine_tune import fine_tune_model
from allennlp.commands.train import train_model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model
from biome.data.utils import configure_dask_cluster
from typing import Optional, Callable

from biome.text.commands.helpers import BiomeConfig
from biome.text.models import load_archive
from biome.text.pipelines.learn.default_callback_trainer import DefaultCallbackTrainer

__alias__ = [DefaultCallbackTrainer]


_logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DATASET_READER_FIELD_NAME = "dataset_reader"
MODEL_FIELD_NAME = "model"


class BiomeLearn(Subcommand):
    def description(self) -> str:
        return "Make a model learn"

    def command_handler(self) -> Callable:
        return learn_from_args

    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
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
            "--workers", type=int, default=1, help="Workers for dask local cluster"
        )

        subparser.add_argument("-v", "--verbose", action="store_true", help="Turn on verbose logs from AllenNLP.")

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
        output=args.output,
        workers=args.workers,
        verbose=args.verbose,
    )


def check_model_configuration(params: Params):
    pass


def learn(
    output: str,
    model_spec: Optional[str] = None,
    model_binary: Optional[str] = None,
    vocab: Optional[str] = None,
    trainer_path: str = "",
    train_cfg: str = "",
    validation_cfg: str = "",
    test_cfg: Optional[str] = None,
    workers: int = 1,
    verbose: bool = False,
) -> Model:

    if verbose:
        logger = logging.getLogger("allennlp")
        logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger("allennlp")
        logger.setLevel(logging.ERROR)

    _logger.info("Starting up learning process.")
    if not model_binary and not model_spec:
        raise ConfigurationError("Missing parameter --spec/--binary")

    allennlp_configuration = BiomeConfig(
        model_path=model_spec,
        trainer_path=trainer_path,
        vocab_path=vocab,
        train_path=train_cfg,
        validation_path=validation_cfg,
        test_path=test_cfg,
    ).to_allennlp_params()

    _logger.info("Launching dask cluster")
    client = configure_dask_cluster(n_workers=workers)

    # Vocabulary is needed for components instantiation
    # TODO: Include a proper checking of the model configuration
    # _logger.info("Checking model configuration")
    # check_model_configuration(Params(deepcopy(allennlp_configuration)))
    try:
        allennlp_configuration = allennlp_configuration.copy()
        if model_binary:
            archive = load_archive(model_binary)
            _logger.info(f"Loading '{MODEL_FIELD_NAME}' config: {archive.config.as_dict()[MODEL_FIELD_NAME]}")
            _logger.info(
                f"Loading '{DATASET_READER_FIELD_NAME}' config: {archive.config.as_dict()[DATASET_READER_FIELD_NAME]}"
            )
            _logger.info(f"Provided configs: {allennlp_configuration}")
            # The model params are ignored by the `fine_tune_model` method
            fine_tune_params = Params(
                {
                    DATASET_READER_FIELD_NAME: archive.config.get(
                        DATASET_READER_FIELD_NAME
                    ).as_dict(),
                    MODEL_FIELD_NAME: archive.config.get(
                        MODEL_FIELD_NAME
                    ).as_dict(),
                    **allennlp_configuration,
                }
            )
            # Force clean folder for run fine tuning properly
            shutil.rmtree(output, ignore_errors=True)

            return fine_tune_model(
                model=archive.model,
                params=fine_tune_params,
                serialization_dir=output,
                extend_vocab=True,
                file_friendly_logging=True,
            )
        else:
            params = Params(allennlp_configuration)
            is_recovered = recover_output_folder(output, params)
            return train_model(
                params=params,
                serialization_dir=output,
                file_friendly_logging=True,
                recover=is_recovered,
            )
    finally:
        client.close()


def recover_output_folder(output: str, params: Params) -> bool:
    """If output folder already exists, we automatically recover the generated vocab in this folder.

    Allows reuse the generated vocab if something went wrong in previous executions

    Parameters
    ----------
    output
        Path to the output folder
    params
        Parameters for the train command

    Returns
    -------
    is_recovered
        True if existing output folder is recovered, False if output folder does not exist.
    """
    if not os.path.isdir(output):
        return False
    else:
        [
            os.remove(file)
            for pattern in [
                os.path.join(output, "*.th"),
                os.path.join(output, "*.json"),
                os.path.join(output, "**/events.out*"),
            ]
            for file in glob.glob(pattern, recursive=True)
        ]
        params.to_file(os.path.join(output, CONFIG_NAME))
        _logger.warning(f"Using vocab from recovered output folder '{output}' if available.")

        return True
