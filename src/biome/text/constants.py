import os

from .environment import BIOME_EXPLORE_ENDPOINT

BIOME_METADATA_INDEX = ".biome"
# This is the biome explore UI endpoint, used for show information
# about explorations once the data is persisted
EXPLORE_APP_ENDPOINT = os.getenv(BIOME_EXPLORE_ENDPOINT, "http://localhost:8080")
