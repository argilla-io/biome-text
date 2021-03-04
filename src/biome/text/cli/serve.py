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
@click.option(
    "--host",
    "-h",
    type=str,
    default="0.0.0.0",
    help="Host of the underlying uvicorn server",
)
def serve(pipeline_path: str, port: int, predictions_dir: str, host: str) -> None:
    """Serves the pipeline predictions as a REST API

    PIPELINE_PATH is the path to a pretrained pipeline (model.tar.gz file).
    """
    pipeline = Pipeline.from_pretrained(pipeline_path)
    pipeline._model.eval()

    if predictions_dir:
        pipeline.init_prediction_logger(predictions_dir)

    return _serve(pipeline, port, host)


def _serve(pipeline: Pipeline, port: int = 9999, host: str = "0.0.0.0"):
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

        @app.post("/predict", tags=["Pipeline"])
        async def predict(predict_input: PredictInput):
            """Returns a prediction given some input data

            Parameters
            ----------
            - **args/kwargs:** See the Example Value for the Request body below.
            If provided, the **batch** parameter will be ignored.
            - **batch:** A list of dictionaries that represents a batch of inputs.
            The dictionary keys must comply with the **args/kwargs**.
            Predicting batches should typically be faster than repeated calls with **args/kwargs**.
            - **add_tokens:** If true, adds a 'tokens' key in the prediction that contains the tokenized input.
            - **add_attributions:** If true, adds a 'attributions' key that contains attributions of the input to the prediction.
            - **attributions_kwargs:** This dict is directly passed on to the `TaskHead.compute_attributions()`.

            Returns
            -------
            - **predictions:** A dictionary or a list of dictionaries containing the predictions and additional information.
            """
            with http_error_handling():
                return sanitize(
                    pipeline.predict(**predict_input.dict(skip_defaults=True))
                )

        @app.get("/config", tags=["Pipeline"])
        async def config():
            """The configuration of the pipeline"""
            with http_error_handling():
                return pipeline.config.as_dict()

        @app.get("/_status", tags=["REST service"])
        async def status():
            with http_error_handling():
                return {"ok": True}

        return app

    uvicorn.run(make_app(), host=host, port=port)
