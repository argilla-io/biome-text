import argparse
import datetime
import logging
import time
from typing import Dict, Iterable, Any, List

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from biome.allennlp.models import to_local_archive
from biome.allennlp.predictors.utils import get_predictor_from_archive
from biome.data.sinks import store_dataset
from biome.data.sources import DataSource
from biome.data.utils import configure_dask_cluster, default_elasticsearch_sink

# TODO centralize configuration
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class BiomePredict(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = """Make a batch prediction over input test data set"""

        subparser = parser.add_parser(
            name,
            description=description,
            help="Use a trained model to make predictions.",
        )

        subparser.add_argument(
            "--binary",
            type=str,
            help="the archived model to make predictions with",
            required=True,
        )
        subparser.add_argument(
            "--from-source",
            type=str,
            help="datasource source definition",
            required=True,
        )
        subparser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="The batch size to use for processing",
        )
        subparser.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )
        subparser.add_argument(
            "--workers", type=int, default=1, help="Workers for dask local cluster"
        )

        subparser.set_defaults(func=_predict)

        return subparser


def __predictor_from_args(
    archive_file: str, cuda_device: int, include_package: List[str] = []
) -> Predictor:
    for package_name in include_package:
        import_submodules(package_name)

    archive = load_archive(archive_file, cuda_device=cuda_device)

    # Matching predictor name with model name
    return get_predictor_from_archive(archive)


def _predict(args: argparse.Namespace) -> None:
    predict(
        binary=args.binary,
        from_source=args.from_source,
        cuda_device=args.cuda_device,
        workers=args.workers,
    )


def predict(
    binary: str,
    from_source: str,
    workers: int = 1,
    worker_mem: int = 2e9,
    batch_size: int = 1000,
    cuda_device: int = -1,
    to_sink: dict = None,
) -> str:
    """

    Parameters
    ----------
    binary
    from_source
    workers
    worker_mem
    batch_size
    cuda_device

    Returns
    -------
    index
        Name of the Elasticsearch index where the predictions are stored
    """

    logging.getLogger("allennlp.common.params").setLevel(logging.WARNING)
    logging.getLogger("allennlp.common").setLevel(logging.WARNING)

    def predict_partition(partition: Iterable) -> List[Dict[str, Any]]:
        predictor = __predictor_from_args(
            archive_file=to_local_archive(binary), cuda_device=cuda_device
        )
        return predictor.predict_batch_json(partition)

    prediction_start_time = time.time()

    data_source = DataSource.from_yaml(from_source)
    sink_config = (
        default_elasticsearch_sink(from_source, binary, batch_size)
        if not to_sink
        else to_sink
    )

    configure_dask_cluster(n_workers=workers, worker_memory=worker_mem)
    test_dataset = data_source.read(include_source=True).persist()
    npartitions = max(1, round(test_dataset.count().compute() / batch_size))

    predicted_dataset = (
        test_dataset.repartition(npartitions=npartitions)
        .map_partitions(predict_partition)
        .persist()
    )

    [_logger.info(result) for result in store_dataset(predicted_dataset, sink_config)]

    prediction_elapsed_time = time.time() - prediction_start_time
    formatted_time = str(datetime.timedelta(seconds=int(prediction_elapsed_time)))
    _logger.info("Prediction and indexing time: %s", formatted_time)

    return sink_config["index"]
