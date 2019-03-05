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

## Getting started
The best way to get started is to check our get started Github project [Biome classifiers]() which gives you access to pre-trained, pre-configured state of the art classification models. 

You can also read our article ["Introducing Recognai Biome: learn, predict, explore...(TBD)""]() on Medium.

Biome-text can be installed as a Python library using `pip`:

```bash
pip install https://github.com/recognai/biome-text.git
```

## Working with Biome: Learn, predict, explore
Biome-text has a very similar workflow to AllenNLP, extending existing commands and defining new ones. The main commands are:

### Learn

```bash
biome learn
    --spec=models/bidirectional_rnns/bi.gru.cased.yml 
    --train=datasources/ag_news_train.yml 
    --validation=datasources/ag_news_test.yml 
    --output=bi.gru.cased.adam 
    --trainer=trainers/basic.adam.2.yml
```

### Predict

```bash
biome predict
    --binary=bi.gru.cased.adam/model.tar.gz
```

### Explore
```bash
biome explore
```


## Running and exploring your first model

## Motivation
At [Recognai](http://recogn.ai), we love and have been working (and sometimes contributing to) libraries like [spaCy](http://spacy.io), [PyTorch](http://pytorch.org) and [AllenNLP](https://allennlp.org) since their first releases. 

Biome has been an internal tool for making it easier to work on use cases working with large datasets, changing requirements, evolving models, and especially for **working and communicating with non-technical people and teams**. 

We decided to release Biome with the hope that is useful to others, from more advanced NLP developers to people interested in getting started with these technologies. 

Biome relies heavily in and is fully compatible with AllenNLP. 

You can use Biome as Python package in your AllenNLP's workflows and/or use Biome extended functionalities with your AllenNLP models.

## Contributing
If you'd like to contribute, please read our contributing guidelines.

## Roadmap

* Include sequence labelling models such as Named entity recognition models.


## Licensing

The code in this project is licensed under Apache 2 license.




