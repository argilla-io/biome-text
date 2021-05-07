# Developer guides

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

## Make a release

To make a release you have to follow 4 steps:

1. Run the `prepare_versioned_build.sh` script inside the `docs` folder and commit the changes to the master branch.
The commit message should say something like: "v2.2.0 release".

2. Create a new [GitHub release](https://docs.github.com/en/free-pro-team@latest/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release).

    The version tags should be `v1.1.0` or for release candidates `v1.1.0rc1`.
    Major and minor releases should always be made against the master branch, bugfix releases against the corresponding minor release tag.

    After publishing the release, the CI is triggered and if everything goes well the release gets published on PyPi.
    The CI does:
    - run tests & build docs
    - build package
    - upload to testpypi
    - install from testpypi
    - upload to pypi

3. Revert the last commit in which you changed the docs, the commit message should read something like:
"back to master release".

4. **Docs**: In order for the Algolia Search to work, you need to add the new version number of the docs to our
algolia [config file](https://github.com/algolia/docsearch-configs/blob/master/configs/recogn_biome-text.json) and submit a PR.


Under the hood the versioning of our package is managed by [`setuptools_scm`](https://github.com/pypa/setuptools_scm),
that basically works with the git tags in a repo.
