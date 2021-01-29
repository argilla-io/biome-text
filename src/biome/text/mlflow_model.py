import pandas as pd

from biome.text import Pipeline


class BiomeTextModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict(self, dataframe: pd.DataFrame):
        batch = dataframe.to_dict(orient="records")
        predictions = self.pipeline.predict(batch=batch)
        # covering the case when batch size is 1
        if not isinstance(predictions, list):
            predictions = [predictions]

        return pd.DataFrame(predictions)


def _load_pyfunc(path: str):
    pipeline = Pipeline.from_pretrained(path)

    return BiomeTextModel(pipeline)
