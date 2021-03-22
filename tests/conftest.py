import os
from pathlib import Path

import pytest


def pytest_configure(config):
    # It's really hard to do testing with wandb enabled ...
    os.environ["WANDB_MODE"] = "disabled"


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


@pytest.fixture
def configurations_path() -> Path:
    repo_root = Path(__file__).parent.parent
    return (
        repo_root
        / "docs"
        / "docs"
        / "documentation"
        / "user-guides"
        / "2-configuration.md"
    )


@pytest.fixture
def change_to_tmp_working_dir(tmp_path) -> Path:
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)
