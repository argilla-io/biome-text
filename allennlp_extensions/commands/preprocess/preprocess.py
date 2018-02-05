import argparse
import json
import logging

import os

import sys
from copy import deepcopy

import dill
import torch
from allennlp.commands import Subcommand
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.tee_logger import TeeLogger
from allennlp.data import DatasetReader, Vocabulary, DataIterator, Instance
from typing import Dict, Iterable
from allennlp_extensions.data.dataset import save_to_file

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Preprocess(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Launch a pipeline for dataset preprocessing '''

        subparser = parser.add_parser(name, description=description, help='Preprocess datasets')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the pipeline to be executed')

        # This is necessary to preserve backward compatibility
        serialization = subparser.add_mutually_exclusive_group(required=True)
        serialization.add_argument('-s', '--serialization-dir',
                                   type=str,
                                   help='output folder')
        serialization.add_argument('--serialization_dir',
                                   type=str,
                                   help=argparse.SUPPRESS)

        subparser.set_defaults(func=preprocess_from_args)

        return subparser


def preprocess_from_args(args: argparse.Namespace):
    preprocess_from_file(args.param_path, args.serialization_dir)


def preprocess_from_file(parameter_filename: str, serialization_dir: str) -> None:
    params = Params.from_file(parameter_filename)
    preprocess(params, serialization_dir)


def preprocess(params: Params, serialization_dir: str):
    # TODO move this function to train commmand
    # prepare_environment(params)

    prepare_serialization_dir(params, serialization_dir)

    # Now we begin assembling the required parts for the Trainer.
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    all_datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        all_datasets["validation"] = validation_data
    else:
        validation_data = None

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = dataset_reader.read(test_data_path)
        all_datasets["test"] = test_data
    else:
        test_data = None

    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))

    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   [instance for key, dataset in all_datasets.items()
                                    for instance in dataset
                                    if key in datasets_for_vocab_creation])

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    # TODO save collection with pickle

    # TODO Ideally save_to_file and load_from_file should be defined for each object type. But for now, we keep as an util
    save_to_file(list(train_data), os.path.join(serialization_dir, "train.data"))
    # TODO: what happens when validation data is None?
    save_to_file(list(validation_data), os.path.join(serialization_dir, "validation.data"))
    # TODO: @frascuchon we need to handle test data as well.abs


def prepare_serialization_dir(params, serialization_dir):
    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout,
                           file_friendly_terminal_output=False)  # type: ignore
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr,
                           file_friendly_terminal_output=False)  # type: ignore

    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)

    serialization_params = deepcopy(params).as_dict(quiet=True)

    with open(os.path.join(serialization_dir, "preprocess.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)
