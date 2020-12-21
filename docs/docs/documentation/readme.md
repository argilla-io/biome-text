# Installation

For the installation we recommend setting up a fresh [conda](https://docs.conda.io/en/latest/miniconda.html) environment:

```shell script
conda create -n biome python~=3.7.0 pip>=20.3.0
conda activate biome
```

Once the conda environment is activated, you can install the latest release or the development version via pip.

## Latest release (recommended)

To install the latest release of *biome.text* type in:

````shell script
pip install -U biome-text
````

After installing *biome.text*, the best way to test your installation is by running the *biome.text* cli command:

```shell script
biome --help
```

For the UI component to work you need a running [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) instance.
We recommend running [Elasticsearch via docker](https://www.elastic.co/guide/en/elasticsearch/reference/7.7/docker.html#docker-cli-run-dev-mode):

````shell script
docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.3.2
````

## Master branch

The *master branch* contains the latest features, but is less well tested.
If you are looking for a specific feature that has not been released yet, you can install the package from our master branch with:

````shell script
pip install -U git+https://github.com/recognai/biome-text.git
````

Be aware that the UI components will not work when installing the package this way.
Check out the [developer guides](community/3-developer_guides.md#setting-up-for-development) on how to build the UI components manually.
