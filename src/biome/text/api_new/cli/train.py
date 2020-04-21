import os
from typing import Optional

import click
from click import Path

from biome.text.api_new import Pipeline, VocabularyConfiguration


@click.command("train", help="Train a pipeline")
@click.argument("pipeline_path", type=Path(exists=True))
@click.option("-o", "--output", "output", type=Path(), required=True)
@click.option("--trainer", "trainer", type=Path(exists=True), required=True)
@click.option("--training", "training", type=Path(exists=True), required=True)
@click.option("--validation", "validation", type=Path(exists=True), required=False)
@click.option("--test", "test", type=Path(exists=True), required=False)
@click.option("-vv", "--verbose", "verbose", is_flag=True, default=False)
def learn(
    pipeline_path: str,
    output: str,
    trainer: str,
    training: str,
    validation: Optional[str] = None,
    test: Optional[str] = None,
    verbose: bool = False,
) -> None:
    _, extension = os.path.splitext(pipeline_path)
    extension = extension[1:].lower()
    pipeline = (
        Pipeline.from_file(
            pipeline_path,
            vocab_config=VocabularyConfiguration(
                sources=[ds for ds in [training, validation, test] if ds]
            ),
        )
        if extension in ["yaml", "yml"]
        else Pipeline.from_binary(pipeline_path)
    )
    pipeline.train(
        output=output,
        trainer=trainer,
        training=training,
        validation=validation,
        test=test,
        verbose=verbose,
    )
