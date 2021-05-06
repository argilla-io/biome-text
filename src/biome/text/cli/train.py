from pathlib import Path
from typing import Optional

import click

from biome.text import Dataset


@click.command()
@click.argument(
    "pipeline_path",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Path of the training output.",
)
@click.option(
    "--trainer",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trainer configuration YAML file.",
)
@click.option(
    "--training",
    type=click.Path(exists=True),
    required=True,
    help="Path to the training data.",
)
@click.option(
    "--validation",
    type=click.Path(exists=True),
    required=False,
    help="Path to the validation data.",
)
@click.option(
    "--test",
    type=click.Path(exists=True),
    required=False,
    help="Path to the test data.",
)
def train(
    pipeline_path: str,
    output: str,
    trainer: str,
    training: str,
    validation: Optional[str] = None,
    test: Optional[str] = None,
) -> None:
    """Train a pipeline.

    PIPELINE_PATH is either the path to a pretrained pipeline (model.tar.gz file),
    or the path to a pipeline configuration (YAML file).
    """
    raise NotImplementedError()


def dataset_from_path(path: str) -> Dataset:
    file_extension = Path(path).suffix
    if file_extension in [".csv"]:
        return Dataset.from_csv(path)
    elif file_extension in [".json", ".jsonl"]:
        return Dataset.from_json(path)
    else:
        raise ValueError(f"Could not create a Dataset from '{path}'")
