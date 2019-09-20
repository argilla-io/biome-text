import json
import os

import requests
from flask import Flask, request, Response, send_file, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix


def make_app(es_host: str, statics_dir: str) -> Flask:
    app = Flask(__name__)  # pylint: disable=invalid-name
    app.wsgi_app = ProxyFix(
        app.wsgi_app
    )  # sets the requester IP with the X-Forwarded-For header
    CORS(app)

    @app.route("/")
    def index() -> Response:  # pylint: disable=unused-variable
        return send_file(os.path.join(statics_dir, "index.html"))

    @app.route("/elastic/<path:index>/_search", methods=["GET", "POST", "OPTIONS"])
    def search_proxy(index: str = None) -> Response:  # pylint: disable=unused-variable
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
        return send_from_directory(statics_dir, path)

    @app.route("/static/js/<path:path>")
    def static_js_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(os.path.join(statics_dir, "static/js"), path)

    @app.route("/static/css/<path:path>")
    def static_css_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(os.path.join(statics_dir, "static/css"), path)

    return app
