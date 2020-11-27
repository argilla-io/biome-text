import click
from click import Path

from biome.text import Pipeline


@click.command()
@click.argument("pipeline_path", type=Path(exists=True))
@click.option(
    "--port",
    "-p",
    type=int,
    default=8888,
    show_default=True,
    help="Port on which to serve the REST API.",
)
@click.option(
    "--predictions_dir",
    "-pd",
    type=click.Path(),
    default=None,
    help="Path to log raw predictions from the service.",
)
def serve(pipeline_path: str, port: int, predictions_dir: str) -> None:
    """Serves the pipeline predictions as a REST API

    PIPELINE_PATH is the path to a pretrained pipeline (model.tar.gz file).
    """
    pipeline = Pipeline.from_pretrained(pipeline_path)

    if predictions_dir:
        pipeline.init_prediction_logger(predictions_dir)

    pipeline.serve(port)
