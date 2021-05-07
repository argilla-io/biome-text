import os
from pathlib import Path
from typing import Optional

import click

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration
from biome.text.helpers import yaml_to_dict


@click.command()
@click.argument(
    "pipeline_path",
    type=click.Path(exists=True),
    required=True,
    help="Either the path to a pretrained pipeline (model.tar.gz file), "
    "or the path to a pipeline configuration (YAML file).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Path of the training output.",
)
@click.option(
    "--trainer_config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trainer configuration YAML file.",
)
@click.option(
    "--train_data",
    type=click.Path(exists=True),
    required=True,
    help="Path to the training data.",
)
@click.option(
    "--valid_data",
    type=click.Path(exists=True),
    required=False,
    help="Path to the validation data.",
)
def train(
    pipeline_path: str,
    output: str,
    trainer_config: str,
    train_data: str,
    valid_data: Optional[str] = None,
) -> None:
    """Train a pipeline.

    PIPELINE_PATH is either the path to a pretrained pipeline (model.tar.gz file),
    or the path to a pipeline configuration (YAML file).
    """
    _, extension = os.path.splitext(pipeline_path)
    extension = extension[1:].lower()
    pipeline = (
        Pipeline.from_yaml(pipeline_path)
        if extension in ["yaml", "yml"]
        else Pipeline.from_pretrained(pipeline_path)
    )

    datasets = {
        "train": dataset_from_path(train_data),
        "validation": dataset_from_path(valid_data) if valid_data else None,
    }

    trainer = Trainer(
        pipeline=pipeline,
        train_dataset=datasets["train"],
        valid_dataset=datasets["validation"],
        trainer_config=TrainerConfiguration(**yaml_to_dict(trainer_config)),
    )
    trainer.fit(output_dir=output)


def dataset_from_path(path: str) -> Dataset:
    file_extension = Path(path).suffix
    if file_extension in [".csv"]:
        return Dataset.from_csv(path)
    elif file_extension in [".json", ".jsonl"]:
        return Dataset.from_json(path)
    else:
        raise ValueError(
            f"Could not create a Dataset from '{path}'. "
            f"We only support following formats: [csv, json, jsonl]"
        )
