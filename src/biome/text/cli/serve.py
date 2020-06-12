import click
from click import Path

from biome.text import Pipeline


@click.command("serve", help="Serves pipeline as rest api service")
@click.argument("pipeline_path", type=Path(exists=True))
@click.option("-p", "--port", "port", type=int, default=8888, show_default=True)
@click.option(
    "--predictions-dir",
    "predictions_dir",
    type=str,
    default=None,
    show_default=True,
    help="Path to log raw predictions from the service.",
)
def serve(pipeline_path: str, port: int, predictions_dir: str) -> None:
    pipeline = Pipeline.from_pretrained(pipeline_path)

    if predictions_dir:
        pipeline.init_prediction_logger(predictions_dir)

    pipeline.serve(port)
