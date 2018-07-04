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
import ujson as json
import logging
from copy import deepcopy
from typing import Optional

import os
import pyhocon
import torch
from allennlp.commands.evaluate import evaluate
from allennlp.commands.train import Train, create_serialization_dir, datasets_from_params
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment, prepare_global_logging
from allennlp.data import Vocabulary, DataIterator
from allennlp.models.archival import CONFIG_NAME, archive_model
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training import Trainer

from recognai.commands.utils import read_datasource_cfg, read_definition_from_model_spec

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TRAIN_DATA_FIELD = 'train_data_path'
VALIDATION_DATA_FIELD = 'validation_data_path'
TEST_DATA_FIELD = 'test_data_path'


class RecognaiTrain(Train):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = super(RecognaiTrain, self).add_subparser(name, parser)

        subparser.add_argument('--model-location',
                               required=False,
                               type=str,
                               help='optional location for model binary file (model.tar.gz)')
        subparser.add_argument('--trainer', type=str, help='trainer specs location file', required=True)
        subparser.add_argument('--train', type=str, help='input train source definition', required=True)
        subparser.add_argument('--validation', type=str, help='input validation source definition', required=False)
        subparser.add_argument('--test', type=str, help='input test source definition', required=False)
        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    train_model_from_file(
        args.param_path,
        args.serialization_dir,
        args.model_location,

        args.trainer,

        args.train,
        args.validation,
        args.test,

        args.overrides
    )


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          model_location: Optional[str] = None,

                          trainer_path: str = '',

                          train_cfg: str = "",
                          validation_cfg: str = "",
                          test_cfg: Optional[str] = None,

                          overrides: str = "",
                          vocab_path: str = None) -> Model:
    with open(trainer_path) as trainer_file:
        cfg_params = read_definition_from_model_spec(parameter_filename)
        trainer_params = json.load(trainer_file)

        config = {
            TRAIN_DATA_FIELD: read_datasource_cfg(train_cfg),
            VALIDATION_DATA_FIELD: read_datasource_cfg(validation_cfg),
            **cfg_params,
            **trainer_params
        }

        if test_cfg and not test_cfg.isspace():
            config.update({TEST_DATA_FIELD: read_datasource_cfg(test_cfg)})

        params = Params(config)
        return train_model(params, serialization_dir, file_friendly_logging=True, model_location=model_location)


# From Allen NLP module
def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False,

                model_location: Optional[str] = None) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    check_for_gpu(params.params.get('trainer').get('cuda_device', -1))

    with open(os.path.join(serialization_dir, CONFIG_NAME), "w") as param_file:
        config_json = json.dumps(params.params, indent=4)
        param_file.write(config_json)

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   (instance for key, dataset in all_datasets.items()
                                    for instance in dataset
                                    if key in datasets_for_vocab_creation))
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model_params = __fetch_model_params(model_location, model_params=params.pop('model'))
    model = Model.from_params(vocab, model_params)

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

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    if test_data and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(best_model, test_data, iterator,
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

    return best_model


def __fetch_model_params(model_location: Optional[str], model_params: Params) -> Params:
    if model_location:
        model_location_config = pyhocon.ConfigFactory.from_dict({"model_location": model_location})
        return Params(model_location_config.with_fallback(pyhocon.ConfigFactory.from_dict(model_params.as_dict())))
    return model_params
