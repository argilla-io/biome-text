from typing import Any
from typing import Dict

import click
import uvicorn
from allennlp.common.util import sanitize
from click import Path
from fastapi import FastAPI

from biome.text import Pipeline
from biome.text.errors import http_error_handling


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
    pipeline._model.eval()

    if predictions_dir:
        pipeline.init_prediction_logger(predictions_dir)

    return _serve(pipeline, port)


def _serve(pipeline: Pipeline, port: int):
    """Serves an pipeline as rest api"""

    def make_app() -> FastAPI:
        app = FastAPI()

        @app.post("/predict")
        async def predict(inputs: Dict[str, Any]):
            with http_error_handling():
                return sanitize(pipeline.predict(**inputs))

        @app.get("/_config")
        async def config():
            with http_error_handling():
                return pipeline.config.as_dict()

        @app.get("/_status")
        async def status():
            with http_error_handling():
                return {"ok": True}

        return app

    uvicorn.run(make_app(), host="0.0.0.0", port=port)
