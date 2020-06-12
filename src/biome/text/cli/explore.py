import re

import click
from click import Path

from biome.text import Pipeline
from biome.text.constants import DEFAULT_ES_HOST
from biome.text.data import DataSource
from biome.text.environment import ES_HOST


def _sanizite_index(index_name: str) -> str:
    return re.sub(r"\W", "_", index_name)


@click.command(
    "explore", help="Pipeline predictions over a data source for result exploration"
)
@click.argument("pipeline_path", type=Path(exists=True))
@click.option(
    "-ds", "--data-source", "data_source", type=Path(exists=True), required=True
)
@click.option("-e", "--explain", "explain", is_flag=True, default=False)
@click.option("-es", "--es-host", "es_host", envvar=ES_HOST, default=DEFAULT_ES_HOST)
def explore(pipeline_path: str, data_source: str, explain: bool, es_host: str) -> None:
    Pipeline.from_pretrained(pipeline_path).explore(
        data_source=DataSource.from_yaml(data_source), es_host=es_host, explain=explain
    )
