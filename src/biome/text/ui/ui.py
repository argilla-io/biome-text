import logging
import os

from gevent.pywsgi import WSGIServer

from biome.text.environment import ES_HOST

from .app import make_app

__LOGGER = logging.getLogger(__name__)
__LOGGER.setLevel(logging.INFO)

STATICS_DIR = os.path.join(os.path.dirname(__file__), "webapp")


def launch_ui(es_host: str, port: int = 9000) -> None:
    es_host = es_host if es_host else os.getenv(ES_HOST, "http://localhost:9200")

    flask_app = make_app(
        es_host=es_host,
        statics_dir=STATICS_DIR
        # statics_dir=temporal_static_path("classifier")
    )

    http_server = WSGIServer(("0.0.0.0", port), flask_app)
    __LOGGER.info(
        "Running biome UI on %s with elasticsearch backend %s",
        f"http://0.0.0.0:{http_server.server_port}",
        es_host,
    )
    http_server.serve_forever()
