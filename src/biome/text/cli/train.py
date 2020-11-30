import os
from pathlib import Path
from typing import Optional

import click
from elasticsearch import Elasticsearch

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration
from biome.text.helpers import yaml_to_dict


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
    _, extension = os.path.splitext(pipeline_path)
    extension = extension[1:].lower()
    pipeline = (
        Pipeline.from_yaml(pipeline_path)
        if extension in ["yaml", "yml"]
        else Pipeline.from_pretrained(pipeline_path)
    )

    datasets = {
        "train": dataset_from_path(training),
        "validation": dataset_from_path(validation) if validation else None,
        "test": dataset_from_path(test) if test else None,
    }

    pipeline.create_vocabulary(
        VocabularyConfiguration(
            sources=[dataset for dataset in datasets.values() if dataset]
        ),
    )

    pipeline.train(
        output=output,
        trainer=TrainerConfiguration(**yaml_to_dict(trainer)),
        training=datasets["training"],
        validation=datasets["validation"],
        test=datasets["test"],
    )


def dataset_from_path(path: str) -> Dataset:
    file_extension = Path(path).suffix
    if file_extension in [".csv"]:
        return Dataset.from_csv(path)
    elif file_extension in [".json", ".jsonl"]:
        return Dataset.from_json(path)
    # yaml files are used for elasticsearch data
    elif file_extension in [".yaml", ".yml"]:
        from_es_kwargs = yaml_to_dict(path)
        client = Elasticsearch(**from_es_kwargs["client"])
        return Dataset.from_elasticsearch(
            client=client,
            index=from_es_kwargs["index"],
            query=from_es_kwargs.get("query"),
        )
    else:
        raise ValueError(f"Could not create a Dataset from '{path}'")
