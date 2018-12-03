import argparse
import logging
import os

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from typing import Dict, Iterable

from biome.allennlp.models.archival import to_local_archive
from biome.allennlp.predictors.utils import get_predictor_from_archive
from biome.data.helpers import store_dataset
from biome.data.sources.helpers import read_dataset
from biome.data.utils import configure_dask_cluster
from biome.data.utils import read_datasource_cfg
from biome.helpers import create_es_runner

__logger = logging.getLogger(__name__)


class BiomePredict(Subcommand):

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Make a batch prediction over input test data set'''

        subparser = parser.add_parser(name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('--binary', type=str, help='the archived model to make predictions with', required=True)
        subparser.add_argument('--from-source', type=str, help='datasource source definition', required=True)
        subparser.add_argument('--to-sink', type=str, help='datasource sink definition', default=None)
        subparser.add_argument('--batch-size', type=int, default=1000, help='The batch size to use for processing')
        subparser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.set_defaults(func=_predict)

        return subparser


def _get_predictor(args: argparse.Namespace) -> Predictor:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    # Matching predictor name with model name
    return get_predictor_from_archive(archive)


def __predict(partition: Iterable[Dict], args: argparse.Namespace) -> Iterable[str]:
    for package_name in args.include_package:
        import_submodules(package_name)

    __logger.debug("Creating predictor")
    predictor = _get_predictor(args)
    __logger.debug("Created predictor")

    __logger.debug("batching prediction")
    results = predictor.predict_batch_json(list(partition))
    __logger.debug("predictions successfully")

    return [predictor.dump_line(output) for model_input, output in zip(partition, results)]


def to_local_elasticsearch(source_config: str, binary_path: str):
    file_name = os.path.basename(source_config)
    model_name = os.path.dirname(binary_path)

    es_runner = create_es_runner()
    return dict(
        index='prediction_{}_with_{}'.format(file_name, model_name),
        type='docs',
        es_hosts='http://localhost:{}'.format(es_runner.es_state.port)
    )


def _predict(args: argparse.Namespace) -> None:
    configure_dask_cluster()

    source_config = read_datasource_cfg(args.from_source)
    if not args.to_sink:
        args.to_sink = to_local_elasticsearch(args.from_source, args.binary)

    sink_config = read_datasource_cfg(args.to_sink)
    test_dataset = read_dataset(source_config, include_source=True)

    source_size = test_dataset.count().compute()
    partitions = max(1, source_size // args.batch_size)
    __logger.debug("Number of partitions {}".format(partitions))

    args.archive_file = to_local_archive(args.binary)

    results = test_dataset.repartition(partitions).map_partitions(__predict, args)
    results = store_dataset(results, sink_config)

    __logger.info(results)
    __logger.info("Finished predictions")
