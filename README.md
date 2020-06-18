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
<p>Natural Language Processing library built with AllenNLP and Huggingface Transformers
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
You can install biome.text with pip or from source.


### Pip
The recommended way of installing the library is using pip. You can install everything required for the library:

```shell
pip install biome-text
```

### Install from Source
To install biome-text from source, clone the repository from github:

````shell
git clone https://github.com/recognai/biome-text.git
cd biome-text
python -m pip install .[testing]
````

Then you must build static web resources for the explore UI to work:
````shell script
cd ui 
npm install 
npm run build
````

*Note: node>=12 is required in your machine. 
You can follow installation instructions [here](https://nodejs.org/en/download/)*

If the `make` command is enabled in your system, you can use `make dev` directive: 

````shell
make dev
````

And `make ui` for building static web resources needed for the explore UI method to work:

````shell script
make ui
````

You can see all defined directives as follow:
````shell script
make help
````

#### Test
Test biome-text with pytest

````shell script
cd biome-text
pytest
````

## Licensing

The code in this project is licensed under Apache 2 license.
