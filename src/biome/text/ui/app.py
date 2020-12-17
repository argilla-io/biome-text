import json
import os
from logging import Logger

import requests
from flask import Flask
from flask import Response
from flask import jsonify
from flask import request
from flask import send_file
from flask import send_from_directory
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

    @app.route(
        "/elastic/"
    )  # Fixes the `WARNING:biome.text.ui.app:Path elastic/ not found` message"
    @app.route(
        "/elastic/<path:path>", methods=["GET", "OPTIONS", "POST", "PUT", "DELETE"]
    )
    def elasticsearch_proxy_handler(path: str = "") -> Response:
        """Handles elasticsearch requests"""
        es_url = f"{es_host}/{path}"

        response = None
        if request.method == "OPTIONS":
            return Response(response="", status=200)
        if request.method == "GET":
            response = requests.get(es_url)
        if request.method == "POST":
            response = requests.post(es_url, json=request.get_json())
        if request.method == "PUT":
            response = requests.put(es_url, json=request.get_json())
        if request.method == "DELETE":
            response = requests.delete(es_url, json=request.get_json())
        return (
            jsonify(json.loads(response.text))
            if response
            else Response(response="Not found", status=404)
        )

    @app.route("/<path:path>")
    def static_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        try:
            return send_from_directory(statics_dir, path)
        except NotFound as error:
            logger.warning("Path %s not found. Error: %s", path, error)
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
