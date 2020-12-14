# Contributing

We are open and very happy to receive contributions to make *biome.text* more useful for you and others.

If you want to start contributing to *biome.text* there are three things you need to do.

1. Create an issue describing the feature you want to work on
2. Setup for development and do the code changes
3. Create a pull request describing your changes

## Creating an issue
You can create a feature request or describe a bug on [Github](https://github.com/recognai/biome-text/issues/new/choose)

## Setting up for development
To set up your system for *biome.text* development, you first of all have to [fork](https://guides.github.com/activities/forking/)
our repository and clone your fork to your computer:

````shell script
git clone https://github.com/[your-github-username]/biome-text.git
cd biome-text
````

Now go ahead and create a new conda environment in which the development will take place and activate it:

````shell script
conda env create -f environment_dev.yml
conda activate biome
````





For a development installation from source see our [installation](../readme.md) section.
After you make changes you can run our formatting and test suite as follows:

````shell script
make test
````

### Running tests locally

### Building docs locally

## Submitting a pull request

For example, a new issue, #13, describing an error found in documentation, and labelled as documentation, you will created an new related branch called documentation/#13

Work on this branch make necessary changes, testing them and then push the new branch and create an new PR.

This new PR will include the text "Closes #13" at the end of the description
