# Contributing

We are open and very happy to receive contributions to make biome.text more useful for you and others.

If you want to start contributing to `biome.text` there are three things you need to do.

To contribute via pull request, follow these steps:

1. Create an issue describing the feature you want to work on
2. Install from source, write your code, tests and documentation, and format them with `black.
3. Create a pull request describing your changes

## 1. Creating an issue
You can create a feature request or describe a bug on [Github](https://github.com/recognai/biome-text/issues/new/choose)

## 2. Installing from source, developing and testing
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

or for a developer installation:
````shell
make dev
````

You can see defined make directives as follows:
````shell script
make help
````

## Testing your changes
After you make changes you can run the tests as follows:

````shell script
cd biome-text
pytest
````

## 3. Submitting a pull request



