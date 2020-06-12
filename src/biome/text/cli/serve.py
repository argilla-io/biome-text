import click
from click import Path

from biome.text import Pipeline


@click.command("serve", help="Serves pipeline as rest api service")
@click.argument("pipeline_path", type=Path(exists=True))
@click.option("-p", "--port", "port", type=int, default=8888, show_default=True)
@click.option("-l", "--log_path", "log_path", type=str, default=None, show_default=True)
def serve(pipeline_path: str, port: int, log_path: str) -> None:
    pipeline = Pipeline.from_pretrained(pipeline_path)

    if log_path:
        pipeline.init_prediction_logger(log_path)

    pipeline.serve(port)
