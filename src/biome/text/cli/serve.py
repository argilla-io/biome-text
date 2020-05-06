import click
from click import Path

from biome.text.pipeline import Pipeline


@click.command("serve", help="Serves pipeline as rest api service")
@click.argument("pipeline_path", type=Path(exists=True))
@click.option("-p", "--port", "port", type=int, default=8888, show_default=True)
def serve(pipeline_path: str, port: int) -> None:
    pipeline = Pipeline.from_pretrained(pipeline_path)
    pipeline.serve(port)
