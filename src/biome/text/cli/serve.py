import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import click
import uvicorn
from allennlp.common.util import sanitize
from click import Path
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from pydantic import create_model
from starlette.exceptions import HTTPException as StarletteHTTPException

from biome.text import Pipeline


@click.command()
@click.argument("pipeline_path", type=Path(exists=True))
@click.option(
    "--port",
    "-p",
    type=int,
    default=9999,
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
    predict_parameters = inspect.signature(pipeline.predict).parameters
    model_parameters = {
        name: (
            par.annotation,
            None,  # We need a default value to allow for batch predictions!
        )
        for name, par in predict_parameters.items()
        if par.default == inspect.Parameter.empty
    }
    optional_parameters = {
        name: (par.annotation, par.default)
        for name, par in predict_parameters.items()
        # The batch parameter needs an extra logic to allow for a proper BaseModel for it
        if par.default != inspect.Parameter.empty and name != "batch"
    }

    class Config:
        extra = "forbid"

    ModelInput = create_model("ModelInput", **model_parameters, __config__=Config)
    PredictInput = create_model(
        "PredictInput",
        **model_parameters,
        batch=(List[ModelInput], None),
        **optional_parameters,
        __config__=Config,
    )

    class http_error_handling:
        """Error handling for http error transcription"""

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            if isinstance(exc_val, Exception):
                # Common http error handling
                raise HTTPException(status_code=500, detail=str(exc_val))

    def make_app() -> FastAPI:
        app = FastAPI()

        error_msg = f"\nCheck the docs at '0.0.0.0:{port}/docs' for an example of a valid request body."

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request, exc):
            return PlainTextResponse(str(exc) + error_msg, status_code=400)

        @app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request, exc):
            if exc.status_code == 400:
                return PlainTextResponse(
                    str(exc.detail) + error_msg, status_code=exc.status_code
                )
            else:
                return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

        @app.post("/predict")
        async def predict(predict_input: PredictInput):
            with http_error_handling():
                return sanitize(
                    pipeline.predict(**predict_input.dict(skip_defaults=True))
                )

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
