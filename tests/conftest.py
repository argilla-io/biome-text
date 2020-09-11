from biome.text import loggers


def pytest_configure(config):
    # In case you have wandb installed, there is an issue with tests:
    # https://github.com/wandb/client/issues/1138
    loggers._HAS_WANDB = False
