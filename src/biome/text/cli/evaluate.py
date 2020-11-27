from typing import Optional

import click

from biome.text import Pipeline
from biome.text.cli.train import dataset_from_path


@click.command()
@click.argument(
    "pipeline_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Path to write the evaluation metrics to.",
)
@click.option(
    "--dataset",
    "-ds",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset",
)
@click.option(
    "--batch_size",
    "-bs",
    type=int,
    default=16,
    show_default=True,
    help="Batch size during evaluation.",
)
@click.option(
    "--lazy",
    "-l",
    type=bool,
    default=False,
    show_default=True,
    help="If true, data is lazily loaded from disk, otherwise it is loaded into memory.",
)
@click.option(
    "--prediction_output",
    "-po",
    type=click.Path(),
    default=None,
    help="Write batch predictions to this file.",
)
def evaluate(
    pipeline_path: str,
    output: str,
    dataset: str,
    batch_size: int = 16,
    lazy: bool = False,
    prediction_output: Optional[str] = None,
) -> None:
    """Evaluate a pipeline on a given dataset.

    PIPELINE_PATH is the path to a pretrained pipeline (model.tar.gz file).
    """
    pipeline = Pipeline.from_pretrained(pipeline_path)
    dataset = dataset_from_path(dataset)

    pipeline.evaluate(
        dataset,
        batch_size=batch_size,
        lazy=lazy,
        predictions_output_file=prediction_output,
        metrics_output_file=output,
    )
