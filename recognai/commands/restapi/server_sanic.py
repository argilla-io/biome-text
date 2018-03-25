"""
A `Sanic <http://sanic.readthedocs.io/en/latest/>`_ server that serves up
AllenNLP models as well as our demo.

Usually you would use :mod:`~allennlp.commands.serve`
rather than instantiating an ``app`` yourself.
"""
import json
import logging
from typing import Dict

import os
from allennlp.common.util import JsonDict
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from functools import lru_cache
from sanic import Sanic, response, request
from sanic.exceptions import ServerError

# Can override cache size with an environment variable. If it's 0 then disable caching altogether.
CACHE_SIZE = os.environ.get("SANIC_CACHE_SIZE") or 128

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def run(port: int, workers: int,
        trained_models: Dict[str, str],
        static_dir: str = None) -> None:
    """Run the server programatically"""
    print("Starting a sanic server on port {}.".format(port))

    app = make_app(static_dir)

    for predictor_name, archive_file in trained_models.items():
        archive = load_archive(archive_file)
        predictor = Predictor.from_archive(archive, predictor_name)

        app.model = predictor

    app.run(port=port, host="0.0.0.0", workers=workers)


def make_app(build_dir: str = None) -> Sanic:
    app = Sanic(__name__)  # pylint: disable=invalid-name
    app.model = None

    try:
        cache_size = int(CACHE_SIZE)  # type: ignore
    except ValueError:
        logger.warning("unable to parse cache size %s as int, disabling cache", CACHE_SIZE)
        cache_size = 0

    @lru_cache(maxsize=cache_size)
    def _caching_prediction(model: Predictor, data: str) -> JsonDict:
        """
        Just a wrapper around ``model.predict_json`` that allows us to use a cache decorator.
        """
        return model.predict_json(json.loads(data))

    @app.route('/predict', methods=['POST'])
    async def predict(req: request.Request,
                      model_name: str) -> response.HTTPResponse:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        model = app.model

        data = req.json
        log_blob = {"inputs": data, "cached": False, "outputs": {}}

        # See if we hit or not. In theory this could result in false positives.
        pre_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        try:
            if cache_size > 0:
                # lru_cache insists that all function arguments be hashable,
                # so unfortunately we have to stringify the data.
                prediction = _caching_prediction(model, json.dumps(data))
            else:
                # if cache_size is 0, skip caching altogether
                prediction = model.predict_json(data)
        except KeyError as err:
            raise ServerError("Required JSON field not found: " + err.args[0], status_code=400)

        post_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        if post_hits > pre_hits:
            # Cache hit, so insert an artifical pause
            log_blob["cached"] = True

        # The model predictions are extremely verbose, so we only log the most human-readable
        logger.info("prediction: %s", json.dumps(log_blob))
        return response.json(prediction)

    return app
