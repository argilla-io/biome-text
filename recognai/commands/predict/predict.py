import argparse
import logging
import ujson as json
from typing import Dict, Iterable

import math
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

from recognai.data.sources.helpers import read_dataset
from recognai.data.sinks.helpers import store_dataset

__logger = logging.getLogger(__name__)


class Predict(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Make a batch prediction over input test data set'''

        subparser = parser.add_parser(name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('--from-source', type=str, help='input source definition', required=True)
        subparser.add_argument('--to-sink', type=str, help='output sink definition', required=True)
        subparser.add_argument('--weights-overrides', type=str, help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1000, help='The batch size to use for processing')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('--config-overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict)

        return subparser


def _get_predictor(args: argparse.Namespace) -> Predictor:
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_overrides,
                           cuda_device=args.cuda_device,
                           overrides=args.config_overrides)

    # Predictor explicitly specified, so use it
    return Predictor.from_archive(archive, args.predictor)


def __predict(partition: Iterable[Dict], args: argparse.Namespace) -> Iterable[str]:
    for package_name in args.include_package:
        import_submodules(package_name)

    __logger.info("Creating predictor")
    predictor = _get_predictor(args)
    __logger.info("Created predictor")

    __logger.info("batching prediction")
    # results = [predictor.predict_json(example, args.cuda_device) for example in partition]
    results = predictor.predict_batch_json(partition, args.cuda_device)
    __logger.info("predictions successfully")

    return [predictor.dump_line(output) for model_input, output in zip(partition, results)]


def _predict(args: argparse.Namespace) -> None:
    source_config = json.loads(args.from_source)
    sink_config = json.loads(args.to_sink)
    test_dataset = read_dataset(source_config)

    source_size = test_dataset.count().compute()
    partitions = max(1, source_size // args.batch_size)
    __logger.info("Number of partitions {}".format(partitions))

    results = test_dataset.repartition(partitions).map_partitions(__predict, args)
    results = store_dataset(results, sink_config)

    __logger.info(results)
    __logger.info("Finished predictions")
