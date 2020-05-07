import logging
import os
import tarfile
from tempfile import mkdtemp
from typing import Optional

from gevent.pywsgi import WSGIServer

from biome.text.environment import ES_HOST
from .app import make_app

# TODO centralize configuration
logging.basicConfig(level=logging.INFO)

__LOGGER = logging.getLogger(__name__)


def launch_ui(es_host: str, port: int = 9000) -> None:
    es_host = es_host if es_host else os.getenv(ES_HOST, "http://localhost:9200")

    flask_app = make_app(
        es_host=es_host, statics_dir=temporal_static_path("classifier")
    )

    http_server = WSGIServer(("0.0.0.0", port), flask_app)
    __LOGGER.info(
        "Running biome UI on %s with elasticsearch backend %s",
        f"http://0.0.0.0:{http_server.server_port}",
        es_host,
    )
    http_server.serve_forever()


def temporal_static_path(explore_view: str, basedir: Optional[str] = None):
    statics_tmp = mkdtemp()

    compressed_ui = os.path.join(
        basedir or os.path.dirname(__file__), "{}.tar.gz".format(explore_view)
    )
    tar_file = tarfile.open(compressed_ui, "r:gz")
    tar_file.extractall(path=statics_tmp)
    tar_file.close()

    return statics_tmp
