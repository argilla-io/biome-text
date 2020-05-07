import json
import os
from logging import Logger

import requests
from flask import (
    Flask,
    request,
    Response,
    send_file,
    send_from_directory,
    jsonify,
    redirect,
)
from flask_cors import CORS
from werkzeug.exceptions import NotFound
from werkzeug.middleware.proxy_fix import ProxyFix


def make_app(es_host: str, statics_dir: str) -> Flask:
    app = Flask(__name__)  # pylint: disable=invalid-name
    app.wsgi_app = ProxyFix(
        app.wsgi_app
    )  # sets the requester IP with the X-Forwarded-For header
    CORS(app)

    logger: Logger = app.logger

    @app.route("/elastic/<path:es_path>", methods=["GET", "OPTIONS"])
    def es_info_proxy(es_path: str) -> Response:
        es_url = f"{es_host}/{es_path}"

        if request.method == "OPTIONS":
            return Response(response="", status=200)

        response = requests.get(es_url)
        return jsonify(response.json())

    @app.route("/elastic/<path:es_path>/_update", methods=["POST", "OPTIONS"])
    def es_update_proxy(es_path: str) -> Response:
        es_url = f"{es_host}/{es_path}/_update"
        response = requests.post(es_url, json=request.get_json())
        return jsonify(response.json())

    @app.route("/elastic/<path:index>/_search", methods=["GET", "POST", "OPTIONS"])
    def es_search_proxy(
        index: str = None,
    ) -> Response:  # pylint: disable=unused-variable
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        es_url = (
            "{}/{}/_search".format(es_host, index)
            if index
            else "{}/_search".format(es_host)
        )

        if request.method == "GET":
            response = requests.get(es_url)
        else:
            response = requests.post(es_url, json=request.get_json())

        return jsonify(json.loads(response.text))

    @app.route("/<path:path>")
    def static_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        try:
            return send_from_directory(statics_dir, path)
        except NotFound as error:
            logger.warning(error)
            return index()

    @app.route("/static/js/<path:path>")
    def static_js_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(os.path.join(statics_dir, "static/js"), path)

    @app.route("/static/css/<path:path>")
    def static_css_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(os.path.join(statics_dir, "static/css"), path)

    @app.route("/public/<path:path>")
    def public_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(statics_dir, path)

    @app.route("/")
    def index() -> Response:  # pylint: disable=unused-variable
        return send_file(os.path.join(statics_dir, "index.html"))

    return app
