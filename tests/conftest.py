from biome.text import loggers
import pytest
from pathlib import Path


def pytest_configure(config):
    # In case you have wandb installed, there is an issue with tests:
    # https://github.com/wandb/client/issues/1138
    loggers._HAS_WANDB = False


@pytest.fixture
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture
def resources_data_path(resources_path) -> Path:
    return resources_path / "data"


@pytest.fixture
def tutorials_path() -> Path:
    repo_root = Path(__file__).parent.parent
    return repo_root / "docs" / "docs" / "documentation" / "tutorials"
