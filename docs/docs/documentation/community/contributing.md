# Contributing

We are open and very happy to receive contributions to make *biome.text* more useful for you and others.

If you want to start contributing to *biome.text* there are three things you need to do.

1. Create an issue describing the feature you want to work on
2. Setup for development and do the code changes
3. Create a pull request describing your changes

## Creating an issue
You can create a feature request or describe a bug on [Github](https://github.com/recognai/biome-text/issues/new/choose).

## Setting up for development
To set up your system for *biome.text* development, you first of all have to [fork](https://guides.github.com/activities/forking/)
our repository and clone your fork to your computer:

````shell script
git clone https://github.com/[your-github-username]/biome-text.git
cd biome-text
````

To keep your fork's master branch up to date with our repo you should add it as an [upstream remote branch](https://dev.to/louhayes3/git-add-an-upstream-to-a-forked-repo-1mik):

````shell script
git remote add upstream https://github.com/recognai/biome-text.git
````

Now go ahead and create a new conda environment in which the development will take place and activate it:

````shell script
conda env create -f environment_dev.yml
conda activate biome
````

Once you activated the conda environment, it is time to install *biome.text* in editable mode with all its development dependencies.
The best way to do this is to take advantage of the make directive:

````shell script
make dev
````

After installing *biome.text*, the best way to test your installation is by running the *biome.text* cli command:

```shell script
biome --help
```

### Building the UI components

For the UI to work you need to build the static web resources:

````shell script
make build_ui
````

If you are working on the UI and want to quickly check out the results you can serve it with:

```shell script
make ui
```

Keep in mind that for the UI component to work you need a running [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) instance.
We recommend running [Elasticsearch via docker](https://www.elastic.co/guide/en/elasticsearch/reference/7.7/docker.html#docker-cli-run-dev-mode):

````shell script
docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.3.2
````

### Running tests locally

*Biome.text* uses [pytest](https://docs.pytest.org/en/latest/) for its unit and integration tests.
If you are working on the code base we advise you to run our tests locally before submitting a Pull Request (see below) to make sure your changes did not break and existing functionality.
To achieve this you can simply run:

````shell script
make test
````

If you open a Pull Request, the test suite will be run automatically via a GitHub Action.

### Serving docs locally

If you are working on the documentation and want to check out the results locally on your machine, you can simply run:

````shell script
make docs
````

The docs will be built and deployed automatically via a GitHub Action when our master branch is updated.
If for some reason you want to build them locally, you can do so with:

````shell script
make build_docs
````

## Submitting a Pull Request

For example, a new issue, #13, describing an error found in documentation, and labelled as documentation, you will created an new related branch called documentation/#13

Work on this branch make necessary changes, testing them and then push the new branch and create an new PR.

This new PR will include the text "Closes #13" at the end of the description
