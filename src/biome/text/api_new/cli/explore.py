import re

import click
from click import Path

from biome.text.api_new import Pipeline
from biome.text.api_new.constants import DEFAULT_ES_HOST
from biome.text.api_new.environment import ES_HOST


def _sanizite_index(index_name: str) -> str:
    return re.sub(r"\W", "_", index_name)


@click.command(
    "explore", help="Pipeline predictions over a data source for result exploration"
)
@click.option(
    "-ds", "--data-source", "data_source", type=Path(exists=True), required=True
)
@click.option(
    "-pl", "--pipeline", "pipeline_path", type=Path(exists=True), required=True
)
@click.option("-e", "--explain", "explain", is_flag=True, default=False)
@click.option("-es", "--es-host", "es_host", envvar=ES_HOST, default=DEFAULT_ES_HOST)
def explore(data_source: str, pipeline_path: str, explain: bool, es_host: str) -> None:
    Pipeline.from_binary(pipeline_path).explore(
        ds_path=data_source, es_host=es_host, explain=explain
    )
