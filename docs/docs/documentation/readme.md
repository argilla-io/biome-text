# Installation

You can install biome-text with pip or by installing from source.


## Pip

You can install everything required for library. This is often the right choice:
```shell
pip install biome-text
```

## Install from Source
To install biome-text from source, clone the repository from github:

````shell
git clone https://github.com/recognai/biome-text.git
cd biome-text
python -m pip install .
````

If `make` command is enabled in your system, you can use already defined make directives:

````shell
make install
````  

or 
````shell
make dev
````
for an developer installation

You can see defined directives as follow:
````shell script
make help
````

## Test
Test biome-text with pytest

````shell script
cd biome-text
pytest
````
