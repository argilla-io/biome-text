import argparse
import logging
import os
import re
from typing import Dict, Iterable, Any, List

import dask.bag as db
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

from biome.allennlp.commands.helpers import read_datasource_configuration
from biome.allennlp.models.archival import to_local_archive
from biome.allennlp.predictors.utils import get_predictor_from_archive
from biome.data.sinks import store_dataset
from biome.data.sources.helpers import read_dataset
from biome.data.utils import configure_dask_cluster, default_elasticsearch_sink

__logger = logging.getLogger(__name__)


class BiomePredict(Subcommand):

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Make a batch prediction over input test data set'''

        subparser = parser.add_parser(name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('--binary', type=str, help='the archived model to make predictions with', required=True)
        subparser.add_argument('--from-source', type=str, help='datasource source definition', required=True)
        subparser.add_argument('--to-sink', type=str, help='datasource sink definition', default=None)
        subparser.add_argument('--batch-size', type=int, default=100, help='The batch size to use for processing')
        subparser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.set_defaults(func=_predict)

        return subparser


def __predictor_from_args(args: argparse.Namespace) -> Predictor:
    for package_name in args.include_package:
        import_submodules(package_name)
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    # Matching predictor name with model name
    return get_predictor_from_archive(archive)


def __predict(partition: Iterable[Dict], args: argparse.Namespace) -> Iterable[str]:
    __logger.debug("Creating predictor")
    predictor = __predictor_from_args(args)
    __logger.debug("Created predictor")

    __logger.debug("batching prediction")
    results = predictor.predict_batch_json(list(partition))
    __logger.debug("predictions successfully")

    return [predictor.dump_line(output) for model_input, output in zip(partition, results)]


def _predict(args: argparse.Namespace) -> None:
    configure_dask_cluster(n_workers=1)

    if not args.to_sink:
        args.to_sink = default_elasticsearch_sink(args.from_source, args.binary, args.batch_size)

    source_config = read_datasource_configuration(args.from_source)
    sink_config = read_datasource_configuration(args.to_sink)

    test_dataset = read_dataset(source_config, include_source=True)
    __logger.info("Source sample data:{}".format(test_dataset.take(5)))

    source_size = test_dataset.count().compute()
    batch_size = args.batch_size
    batches = max(1, source_size // batch_size)

    __logger.info("Number of batches {}".format(batches))
    test_dataset = test_dataset.repartition(batches)

    args.archive_file = to_local_archive(args.binary)
    predictor = __predictor_from_args(args)

    for i, batch in enumerate(_get_batch(test_dataset, batch_size)):
        __logger.info('Running prediction batch {}...'.format(i))
        __make_predict(batch, predictor, sink_config)

    __logger.info("Finished predictions")


def _get_batch(dataset, batch_size: int):
    """A batch generator.

    Continuously generates a batch of size `batch_size` out of the provided dataset

    Parameters
    ----------
    dataset : An iterable object
        Batches will be produced out of this dataset
    batch_size : int
        The batch size

    Yields
    ------
    batch : list
        The next batch of the dataset
    """
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def __make_predict(batch: List[Dict[str, Any]], predictor: Predictor, sink_config: Dict[str, Any]):
    results = predictor.predict_batch_json(batch)
    store = db.from_sequence([predictor.dump_line(output) for model_input, output in zip(batch, results)],
                             npartitions=1)
    __logger.info(store_dataset(store, sink_config).persist())
