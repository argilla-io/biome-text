"""Docstring for the mlflow_model.py module

This module is only used by MLFlow internally for loading a Pipeline as MLFlow model from an MLFlow Tracking server.
"""

import pandas as pd

from biome.text import Pipeline


class BiomeTextModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict(self, dataframe: pd.DataFrame):
        batch = dataframe.to_dict(orient="records")
        predictions = self.pipeline.predict(batch=batch)

        return pd.DataFrame(predictions)


def _load_pyfunc(path: str):
    pipeline = Pipeline.from_pretrained(path)

    return BiomeTextModel(pipeline)
