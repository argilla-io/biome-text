import mlflow
import pandas as pd


class BiomeTextModel(mlflow.pyfunc.PythonModel):
    """A custom MLflow model with the 'python_function' flavor for biome.text pipelines.

    This class is used by the `Pipeline.to_mlflow()` method.
    """

    ARTIFACT_CONTEXT = "model"

    def __init__(self):
        self.pipeline = None

    def load_context(self, context):
        from biome.text import Pipeline

        self.pipeline = Pipeline.from_pretrained(
            context.artifacts[self.ARTIFACT_CONTEXT]
        )

    def predict(self, context, dataframe: pd.DataFrame):
        batch = dataframe.to_dict(orient="records")
        predictions = self.pipeline.predict(batch=batch)

        return pd.DataFrame(predictions)
