import re

import click

from biome.text import Pipeline
from biome.text import explore
from biome.text.constants import DEFAULT_ES_HOST
from biome.text.environment import ES_HOST
from .train import dataset_from_path


def _sanizite_index(index_name: str) -> str:
    return re.sub(r"\W", "_", index_name)


@click.command(
    "explore", help="Explore the predictions of a pipeline for a given dataset"
)
@click.argument("pipeline_path", type=click.Path(exists=True))
@click.option(
    "-ds", "--dataset", "dataset_path", type=click.Path(exists=True), required=True
)
@click.option("-e", "--explain", "explain", is_flag=True, default=False)
@click.option("-es", "--es-host", "es_host", envvar=ES_HOST, default=DEFAULT_ES_HOST)
def explore(pipeline_path: str, dataset_path: str, explain: bool, es_host: str) -> None:
    pipeline = Pipeline.from_pretrained(pipeline_path)
    dataset = dataset_from_path(dataset_path)
    explore.create(pipeline, dataset, es_host=es_host, explain=explain)
