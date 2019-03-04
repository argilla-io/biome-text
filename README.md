# README #

AllenNLP extensions contains custom extensions for library [allenNLP](http://allennlp.org/)

### Extensions

* Readers
* Models
* Shell commands
* ...

#### Readers

The allennlp readers extension

#### Models

The allennlp models extension

#### Shell commands

The allennlp shell command extensions

* Rest API serve extension
* Kafka topic listen extension

### Install

```
$ git clone https://gitlab.com/recognai-team/biome/biome-allennlp.git
$ cd biome-allennlp
$ pip install .
```

### Development

For development we recommend cloning and installing first `biome-data` with `pip install -e`.
This allows you to develop both modules at the same time.
Otherwise it will be installed as a site-package.

Commands to install `biome-allennlp` with its testing suite:

```
$ git clone https://gitlab.com/recognai-team/biome/biome-allennlp.git
$ cd biome-allennlp
$ pip install -e .[testing]
```

If you do not want to install the packages required for testing, just use:

```
$ pip install -e .
```

