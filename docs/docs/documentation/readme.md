# Installation
You can install *biome.text* with pip or from source.
For the installation we recommend setting up a fresh [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):

```shell
conda create -n biome python==3.7.1
conda activate biome
```

## Pip
The recommended way for installing the library is using pip. You can install everything required for the library as follows:

```shell script
pip install biome-text
```

## From Source
If you want to contribute to *biome.text* you have to install the library from source.
Clone the repository from github:

````shell script
git clone https://github.com/recognai/biome-text.git
cd biome-text
````

and install the library in editable mode together with the test dependencies:

```shell script
pip install --upgrade -e .[testing]
```

If the `make` command is enabled in your system, you can also use the `make dev` directive:

````shell script
make dev
````

For the UI to work you need to build the static web resources:
````shell script
cd ui
npm install
npm run build
````

*Note: node>=12 is required in your machine.
You can follow installation instructions [here](https://nodejs.org/en/download/)*

Again, you can also use the `make ui` directive if the `make` command is enabled in your system:

````shell script
make ui
````

You can see all defined directives with:
````shell script
make help
````

After installing *biome.text*, the best way to test your installation is by running the *biome.text* cli command:

```shell script
biome --help
```

## Tests
*Biome.text* uses [pytest](https://docs.pytest.org/en/latest/) for its unit and integration tests.
To run the tests, make sure you installed *biome.text* together with its test dependencies and simply execute pytest from within the `biome-text` directory:

````shell script
cd biome-text
pytest
````

## Docs

To build the documentation locally you need to first install *biome.text* together with `pdoc3~=0.8.1`.
pdoc3~=0.8.1
