<p align="center">
    <br>
    <img src="https://github.com/recognai/biome-text/raw/master/docs/biome_text_logo_for_readme.png" width="600"/>
    <br>
<p>
<p align="center">
    <a href="https://travis-ci.org/recognai/biome-text">
        <img alt="Build" src="https://travis-ci.org/recognai/biome-text.svg?branch=master">
    </a>
    <a href="https://github.com/recognai/biome-text/blob/master/LICENSE.txt">
        <img alt="GitHub" src="https://img.shields.io/github/license/recognai/biome-text.svg?color=blue">
    </a>
    <a href="https://www.recogn.ai/biome-text/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/www.recogn.ai/biome-text/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/recognai/biome-text/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/recognai/biome-text.svg">
    </a>
</p>

<h3 align="center">
<p>Natural Language Processing library built with AllenNLP
</h3>

## Quick Links
- [Documentation](https://www.recogn.ai/biome-text/documentation/)


## Features
* State-of-the-art and not so state-of-the-art models trained with **your own data** with simple workflows.

* **Exploration UI** for error analysis with interpretations.

* **Efficient data reading** for (large) datasets in multiple formats and sources (CSV, Parquet, JSON, Elasticsearch, etc.).

* **Modular configuration and extensibility** of models, datasets and training runs programmatically or via config files.

* Use via **`cli`** or as plain Python (e.g., inside a Jupyter Notebook)

* **Compatible with AllenNLP**

## Installation
You can install *biome.text* with pip or from source.
For the installation we recommend setting up a fresh [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):

```shell
conda create -n biome python==3.7.1
conda activate biome
```

### Pip
The recommended way for installing the library is using pip. You can install everything required for the library as follows:

```shell script
pip install biome-text
```

### Install from Source
To install **biome.text* from source, clone the repository from github:

````shell script
git clone https://github.com/recognai/biome-text.git
cd biome-text
pip install .
````

For the UI to work you need to build the static web resources:
````shell script
cd ui 
npm install 
npm run build
````

*Note: node>=12 is required in your machine. 
You can follow installation instructions [here](https://nodejs.org/en/download/)*

For development you should install *biome.text* in the editable mode together with the test dependencies:

```shell script
cd biome-text
pip install --upgrade -e .[testing]
```

If the `make` command is enabled in your system, you can instead use the `make dev` directive for a development installation:

````shell script
make dev
````

and the `make ui` directive for building the static web resources:

````shell script
make ui
````

You can see all defined directives with:
````shell script
make help
````

#### Tests
*Biome.text* uses [pytest](https://docs.pytest.org/en/latest/) for its unit and integration tests.
To run the tests, make sure you installed *biome.text* together with its test dependencies and simply execute pytest from within the `biome-text` directory:

````shell script
cd biome-text
pytest
````

## Licensing

The code in this project is licensed under Apache 2 license.
