<p align="center">
    <br>
    <img src="https://github.com/recognai/biome-text/raw/master/docs/biome_text_logo_for_readme.png" width="600"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/recognai/biome-text/actions">
        <img alt="CI" src="https://github.com/recognai/biome-text/workflows/CI/badge.svg?branch=master&event=push">
    </a>
    <a href="https://codecov.io/gh/recognai/biome-text">
        <img src="https://codecov.io/gh/recognai/biome-text/branch/master/graph/badge.svg?token=943463Z6F7"/>
    </a>
    <a href="https://github.com/recognai/biome-text/blob/master/LICENSE.txt">
        <img alt="GitHub" src="https://img.shields.io/github/license/recognai/biome-text.svg?color=blue">
    </a>
    <a href="https://www.recogn.ai/biome-text/">
        <img alt="Documentation" src="https://img.shields.io/website/http/recognai.github.io/biome-text/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/recognai/biome-text/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/recognai/biome-text.svg">
    </a>
</p>

<h3 align="center">
<p>Natural Language Processing library built with AllenNLP
</h3>

## Quick Links
- [Documentation](https://recognai.github.io/biome-text/)


## Features
* State-of-the-art and not so state-of-the-art models trained with **your own data** with simple workflows.

* **Efficient data reading** for (large) datasets in multiple formats and sources (CSV, Parquet, JSON, etc.).

* **Modular configuration and extensibility** of models, datasets and training runs programmatically or via config files.

* Use via **`cli`** or as plain Python (e.g., inside a Jupyter Notebook)

* **Compatible with AllenNLP**

## Installation

For the installation we recommend setting up a fresh [conda](https://docs.conda.io/en/latest/miniconda.html) environment:

```shell script
conda create -n biome python~=3.7.0 pip>=20.3.0
conda activate biome
```

Once the conda environment is activated, you can install the latest release via pip:

````shell script
pip install -U biome-text
````

After installing *biome.text*, the best way to test your installation is by running the *biome.text* cli command:

```shell script
biome --help
```

## Get started

The best way to see how *biome.text* works is to go through our [first tutorial](https://recognai.github.io/biome-text/master/documentation/tutorials/1-Training_a_text_classifier.html).

Please refer to our [documentation](https://recognai.github.io/biome-text) for more tutorials, detailed user guides and how you can [contribute](https://recognai.github.io/biome-text/master/documentation/community/1-contributing.html) to *biome.text*.

## Licensing

The code in this project is licensed under Apache 2 license.
