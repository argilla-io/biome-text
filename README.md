[![Build Status](https://travis-ci.org/recognai/biome-text.svg?branch=master)](https://travis-ci.org/recognai/biome-text)

# Biome-text
> Biome-text is a light-weight open source Natural Language Processing tool built with AllenNLP

Biome-text gives you an **easy path to state of the art methods** for natural language processing with neural networks. 

## Features
Biome-text complements the excellent library AllenNLP by providing the following features:

* A clean and simple **user interface** for exploring and understanding predictions.

* Test state of the art classifiers with **your own data** in an easy way.

* **Efficient dataset readers** for (large) classification datasets.

* **Modular configuration** for different components such as classification dataset readers, vocabularies, models, and trainers.

* Configurable **text classification models**, including state of the art models such as Google's Bert or AllenAI's Elmo.

* **Fully-compatible with AllenNLP**

## Install
Biome-text supports Python 3.6 and can be installed using pip and conda.

## pip
For installing biome-text with pip, it is highly recommended to install packages in a virtual environment to avoid conflicts with other installations and system packages.

```bash
python -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install https://github.com/recognai/biome-text.git
```

## conda
We provide a conda environment to install most of the package dependencies. We require you to clone this repository and run: 

```bash
conda env create -f environment.yml
conda activate biome
make dev
```

Check the installation by running:

```bash
biome --help
```
You should see the available commands:
```
usage: biome [-h]  ...

Run biome

optional arguments:
  -h, --help  show this help message and exit

Commands:
  
    predict   Use a trained model to make predictions.
    explore   Explore your data
    serve     Run the web service.
    learn     Make a model learn
    vocab     Build a vocabulary
```

<!-- ## Getting started
The best way to get started is to check our get started Github project [Biome classifiers]() which gives you access to pre-trained, pre-configured state of the art classification models. 

You can also read our article ["Introducing Recognai Biome: learn, predict, explore...(TBD)""]() on Medium.

Biome-text can be installed as a Python library using `pip`:

```bash
pip install https://github.com/recognai/biome-text.git
``` -->

## Working with Biome: Learn, predict, explore
Biome-text has a very similar workflow to AllenNLP, extending existing commands and defining new ones.

You can see the available commands and the documentation for each command adding `--help`:
```bash
biome --help
```
You should see the available commands:
```
usage: biome [-h]  ...

Run biome

optional arguments:
  -h, --help  show this help message and exit

Commands:
  
    predict   Use a trained model to make predictions.
    explore   Explore your data
    serve     Run the web service.
    learn     Make a model learn
    vocab     Build a vocabulary
```

### Learn
Basic training command for training models from scratch as well as fine-tuning already trained models (fine-tuning) available as binary tar.gz files.
```bash
biome learn --help
```

```bash
usage: biome learn [-h] [--spec SPEC] [--binary BINARY] [--vocab VOCAB]
                   --trainer TRAINER --train TRAIN [--validation VALIDATION]
                   [--test TEST] --output OUTPUT [--workers WORKERS]
                   [--include-package INCLUDE_PACKAGE]

Make a model learn

optional arguments:
  -h, --help            show this help message and exit
  --spec SPEC           model.yml specification
  --binary BINARY       pretrained model binary tar.gz
  --vocab VOCAB         path to existing vocab
  --trainer TRAINER     trainer.yml specification
  --train TRAIN         train datasource definition
  --validation VALIDATION
                        validation datasource source definition
  --test TEST           test datasource source definition
  --output OUTPUT       learn process generation folder
  --workers WORKERS     Workers for dask local cluster
  --include-package INCLUDE_PACKAGE
```

<!-- Example:

```bash
biome learn
    --spec=models/bidirectional_rnns/bi.gru.cased.yml 
    --train=datasources/ag_news_train.yml 
    --validation=datasources/ag_news_test.yml 
    --output=bi.gru.cased.adam 
    --trainer=trainers/basic.adam.2.yml
``` -->

### Predict
Basic prediction command for using a trained model available as binary tar.gz files to make predictions with a dataset.

```bash
biome predict --help 
```

```bash
usage: biome predict [-h] --binary BINARY --from-source FROM_SOURCE
                     [--batch-size BATCH_SIZE] [--cuda-device CUDA_DEVICE]
                     [--workers WORKERS] [--include-package INCLUDE_PACKAGE]

Make a batch prediction over input test data set

optional arguments:
  -h, --help            show this help message and exit
  --binary BINARY       the archived model to make predictions with
  --from-source FROM_SOURCE
                        datasource source definition
  --batch-size BATCH_SIZE
                        The batch size to use for processing
  --cuda-device CUDA_DEVICE
                        id of GPU to use (if any)
  --workers WORKERS     Workers for dask local cluster
  --include-package INCLUDE_PACKAGE
                        additional packages to include
```

### Explore
```bash
biome explore --help
```

```
usage: biome explore [-h] [--port PORT] [--include-package INCLUDE_PACKAGE]

Explore your data with model annotations

optional arguments:
  -h, --help            show this help message and exit
  --port PORT           Listening port for application
  --include-package INCLUDE_PACKAGE
                        additional packages to include
```


## Motivation
At [Recognai](http://recogn.ai), we love and have been working (and sometimes contributing to) libraries like [spaCy](http://spacy.io), [PyTorch](http://pytorch.org) and [AllenNLP](https://allennlp.org) since their first releases. 

Biome has been an internal tool for making it easier to work on use cases working with large datasets, changing requirements, evolving models, and especially for **working and communicating with professionals and teams not familiarized with data science and natural language processing**. 

We decided to release Biome with the hope that is useful to others, from more advanced NLP developers to people interested in getting started with these technologies. 

Biome relies heavily in and is fully compatible with AllenNLP. 

You can use Biome as Python package in your AllenNLP's workflows and/or use Biome extended functionalities with your AllenNLP models.
<!-- 
## Contributing
If you'd like to contribute, please read our contributing guidelines. -->

## Licensing

The code in this project is licensed under Apache 2 license.

## Setup for development

```
$ git clone https://github.com/recognai/biome-text.git
$ cd biome-text
$ make dev
```

If you do not want to install the packages required for testing, just use:

```
$ pip install -e .
```


