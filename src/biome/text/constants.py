import os

from .environment import BIOME_EXPLORE_ENDPOINT
from .environment import ES_HOST

BIOME_METADATA_INDEX = ".biome"
BIOME_METADATA_INDEX_DOC = "_doc"
# This is the biome explore UI endpoint, used for show information
# about explorations once the data is persisted
EXPLORE_APP_ENDPOINT = os.getenv(BIOME_EXPLORE_ENDPOINT, "http://localhost:8080")

DEFAULT_ES_HOST = os.getenv(ES_HOST, "localhost:9200")
