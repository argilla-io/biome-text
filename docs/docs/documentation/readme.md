
# Installation

You can install the library using `pip` with a virtual environment as follows:
```shell
python -m venv .env
source .env/bin/activate
pip install biome-text
```

After installing biome.text, the best way to test your installation is by running the `biome.text` cli command:
```shell
biome --help
```
And you should see something like:
```
Usage: biome [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  explore  Pipeline predictions over a data source for result exploration
  serve    Serves pipeline as rest api service
  train    Train a pipeline
```

For development installations, contributing or getting help see the [Community](/community/) section.
