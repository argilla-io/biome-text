# Contributing

We are open and very happy to receive contributions to make biome.text more useful for you and others.

If you want to start contributing to `biome.text` there are three things you need to do.

## 1. Installing from source
To install biome.text from source, clone the repository from github:

````shell
git clone https://github.com/recognai/biome-text.git
cd biome-text
python -m pip install .
````

If the `make` command is enabled in your system, you can use already defined make directives:

````shell
make install
````  

or 
````shell
make dev
````
for a developer installation

You can see defined directives as follow:
````shell script
make help
````

## 2. Running tests
Test biome-text with pytest

````shell script
cd biome-text
pytest
````

## 3. Filing issues and pull requests


