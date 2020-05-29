# Contributing

We are open and very happy to receive contributions to make biome.text more useful for you and others.

If you want to start contributing to `biome.text` there are three things you need to do.

To contribute via pull request, follow these steps:

1. Create an issue describing the feature you want to work on
2. Install from source, write your code, tests and documentation, and format them with ``black``
3. Create a pull request describing your changes

## Creating an issue
You can create a feature request or describe a bug on [Github](https://github.com/recognai/biome-text/issues/new/choose)

## Installing from source
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

## Testing
After you make changes you can run the tests as follows:

````shell script
make test
````

## Submitting a pull request

For example, a new issue, #13, describing an error found in documentation, and labelled as documentation, you will created an new related branch called documentation/#13

Work on this branch make necessary changes, testing them and then push the new branch and create an new PR.

This new PR will include the text "Closes #13" at the end of the description



