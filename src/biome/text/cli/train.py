import os
from typing import Optional

import click
from click import Path

from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration
from biome.text.data import DataSource
from biome.text.helpers import yaml_to_dict


@click.command("train", help="Train a pipeline")
@click.argument("pipeline_path", type=Path(exists=True))
@click.option("-o", "--output", "output", type=Path(), required=True)
@click.option("--trainer", "trainer", type=Path(exists=True), required=True)
@click.option("--training", "training", type=Path(exists=True), required=True)
@click.option("--validation", "validation", type=Path(exists=True), required=False)
@click.option("--test", "test", type=Path(exists=True), required=False)
def learn(
    pipeline_path: str,
    output: str,
    trainer: str,
    training: str,
    validation: Optional[str] = None,
    test: Optional[str] = None,
) -> None:
    _, extension = os.path.splitext(pipeline_path)
    extension = extension[1:].lower()
    pipeline = (
        Pipeline.from_yaml(pipeline_path)
        if extension in ["yaml", "yml"]
        else Pipeline.from_pretrained(pipeline_path)
    )

    pipeline.create_vocabulary(
        VocabularyConfiguration(
            sources=[
                DataSource.from_yaml(ds) for ds in [training, validation, test] if ds
            ]
        ),
    )

    pipeline.train(
        output=output,
        trainer=TrainerConfiguration(**yaml_to_dict(trainer)),
        training=training,
        validation=validation,
        test=test,
    )
