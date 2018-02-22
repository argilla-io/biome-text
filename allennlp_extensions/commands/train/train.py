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
from typing import Dict, Iterable
import argparse
import json
import logging
import os
import sys
from copy import deepcopy

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary, Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer
from allennlp_extensions.data.dataset import load_from_file
from allennlp.commands.train import create_serialization_dir
from allennlp.models.archival import CONFIG_NAME

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(
            name, description=description, help='Train a model')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        # This is necessary to preserve backward compatibility
        serialization = subparser.add_mutually_exclusive_group(required=True)
        serialization.add_argument('-s', '--serialization-dir',
                                   type=str,
                                   help='directory in which to save the model and its logs')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('-d', '--vocab_path',
                               type=str,
                               default="",
                               help='path to pre-processed vocabulary')

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path, args.serialization_dir, args.overrides, args.vocab_path)


def train_model_from_file(parameter_filename: str, serialization_dir: str, overrides: str = "",
                          vocab_path: str = None) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, vocab_path)


def build_datasets(params: Params):
    """
    This function builds datasets from a dataset reader and paths definitions.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
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

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = dataset_reader.read(test_data_path)
        all_datasets["test"] = test_data

    return all_datasets


def load_datasets_from_disk(datasets_path: str):
    """
    This function loads preprocessed datasets from a given path

    Parameters
    ----------
    datasets_path: str, required
        The directory in which the datasets are stored
    """
    vocab = Vocabulary.from_files(
        os.path.join(datasets_path, "vocabulary"))

    train_data = load_from_file(os.path.join(datasets_path, 'train.data'))

    validation_data = load_from_file(
        os.path.join(datasets_path, 'validation.data'))
    # TODO: @frascuchon include logic here for test data, once this is added to preprocess command.
    return train_data, validation_data, None, vocab


def build_or_load_datasets(params: Params, serialization_dir: str, vocab_path: str):
    """
    This function builds from a definition or loads preprocessed datasets from a given path

    Parameters
    ----------
    vocab_path: str, required
        The directory in which the vocabulary is stored
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """

    all_datasets = build_datasets(params)
    if vocab_path:
        vocab = Vocabulary.from_files(os.path.join(vocab_path, "vocabulary"))
        # We need to store the vocab in the serialization dir to avoid issues when archiving the models after train/eval
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    else:
        datasets_for_vocab_creation = set(params.pop(
            "datasets_for_vocab_creation", all_datasets))

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(
                    f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))
        vocab = build_vocab(all_datasets, datasets_for_vocab_creation, params, serialization_dir)

    return all_datasets, vocab


def build_vocab(all_datasets, datasets_for_vocab_creation, params, serialization_dir):
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   [instance for key, dataset in all_datasets.items()
                                    for instance in dataset
                                    if key in datasets_for_vocab_creation])

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    return vocab


def train_model(params: Params, serialization_dir: str, vocab_path: str, file_friendly_logging: bool = False) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    prepare_environment(params)
    create_serialization_dir(params, serialization_dir)
    prepare_logging(file_friendly_logging, serialization_dir)

    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, CONFIG_NAME), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    all_datasets, vocab = build_or_load_datasets(params, serialization_dir, vocab_path)

    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    trainer_params = params.pop("trainer")
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    params.assert_empty('base train command')
    metrics = trainer.train()

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    if test_data and evaluate_on_test:
        test_metrics = evaluate(model, test_data, iterator,
                                cuda_device=trainer._cuda_devices[0])  # pylint: disable=protected-access
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    metrics_json = json.dumps(metrics, indent=2)
    with open(os.path.join(serialization_dir, "metrics.json"), "w") as metrics_file:
        metrics_file.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    return model


def prepare_logging(file_friendly_logging, serialization_dir):
    Tqdm.set_slower_interval(file_friendly_logging)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"),  # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"),  # type: ignore
                           sys.stderr,
                           file_friendly_logging)
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
