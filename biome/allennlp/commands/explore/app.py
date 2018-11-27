import json
import logging
import os
import requests

from flask import Flask, request, Response, send_file, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.contrib.fixers import ProxyFix

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_app(es_endpoint: str, statics_dir: str) -> Flask:
    app = Flask(__name__)  # pylint: disable=invalid-name
    app.wsgi_app = ProxyFix(app.wsgi_app)  # sets the requester IP with the X-Forwarded-For header
    CORS(app)

    @app.route('/')
    def index() -> Response:  # pylint: disable=unused-variable
        return send_file(os.path.join(statics_dir, 'index.html'))

    @app.route('/search', methods=['POST', 'OPTIONS'])
    @app.route('/_search', methods=['POST', 'OPTIONS'])
    @app.route('/search/', methods=['POST', 'OPTIONS'])
    @app.route('/_search/', methods=['POST', 'OPTIONS'])
    def search_proxy() -> Response:  # pylint: disable=unused-variable
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        response = requests.post('{}/_search'.format(es_endpoint), json=request.get_json())
        return jsonify(json.loads(response.text))

    @app.route('/<path:path>')
    def static_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(statics_dir, path)

    @app.route('/static/js/<path:path>')
    def static_js_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(os.path.join(statics_dir, 'static/js'), path)

    @app.route('/static/css/<path:path>')
    def static_css_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(os.path.join(statics_dir, 'static/css'), path)

    return app
