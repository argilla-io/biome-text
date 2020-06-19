# The basics
The library is built around a few simple concepts. This section explains everything you need to know to get started.

Before going into details, let's see a simple example:
``` python
from biome.text import Pipeline, VocabularyConfiguration
from biome.text.data import DataSource

pipeline = Pipeline.from_config({
    "name": "my-first-classifier", 
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = DataSource(source='training_data.csv')
pipeline.create_vocabulary(VocabularyConfiguration(sources=[train_ds]))

pipeline.train(
    output="path_to_store_training_run_output",
    training=train_ds
)
``` 

The above example trains a text classifier from scratch by configuring a ``Pipeline``, making a ``Datasource`` from a csv file, and creating a `Vocabulary` with this data. Let's dive into the details.

## Pipeline

Pipelines are the main entry point to the library. A ``Pipeline`` bundles components and actions needed to train, evaluate and use custom NLP models. 

Pipelines encompass tokenization, feature processing, model configuration and actions such as training, serving or inference. 

Let's continue with our example:

``` python{4-7}
from biome.text import Pipeline, VocabularyConfiguration
from biome.text.data import DataSource

pipeline = Pipeline.from_config({
    "name": "my-first-classifier", 
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = DataSource(source='training_data.csv')
pipeline.create_vocabulary(VocabularyConfiguration(sources=[train_ds]))

pipeline.train(
    output="path_to_store_training_run_output",
    training=train_ds
)
```
Here we configure a ``Pipeline`` from scratch using a dictionary. A pipeline can be created in two ways:

1. Using the ``Pipeline.from_config()`` method, which accepts a dict or a `PipelineConfiguration` object.

2. Using YAML configuration files with the `Pipeline.from_yaml()`. The YAML config file in our example would be following:

```yaml
name: my-first-classifier
head:
    type: TextClassification
    labels: ["positive", "negative"]
```
In this example, we only define the `name` and the task we want to train our model on, using the ``head`` parameter, the rest is configured from defaults. In biome.text we try to provide we sensible defaults so you don't have to configure everything just to start experimenting, but there are many things you can tune and configure. 

In particular, a `Pipeline` has the following configurable components:

### Tokenizer
The tokenizer defines how we want to process the text of our input features. Tokenizers are based on [spaCy tokenizers](https://spacy.io/api/tokenizer) and have the following main configuration options:

1. ``lang``: the main language of the text to be tokenized (default is English). Here you can use available [spaCy model codes](https://spacy.io/usage/models/). 
2. ``segment_sentences``: enable sentence splitting for text within your input features, which is especially relevant for long text classification problems.
3. ``text_cleaning``: simple python functions to pre-process text before tokenization. You can define your own but biome.texts provides pre-defined functions for things like cleaning up html tags or remove extra blank spaces.

### Features
Features are a central concept of the library. Building on the flexibility of AllenNLP, biome.text gives you the ability of combining [Word](../api/biome/text/features.md#wordfeatures) and [Character](../api/biome/text/features.md#charfeatures) features easily. There are many things which can be configured here: size of the embeddings, encoder type (e.g., CNNs or RNNs) for character encoding, pre-trained word vectors, and other things.

### Encoder
To support transfer learning, biome.text models have two core components, a model "backbone" and a "task" head. 

The ``Encoder`` is a central piece of the backbone. It's basically a sequence to sequence or seq2seq encoder, which "contextualizes" textual features in the context of a task (supervised or unsupervised). In this way, the encoder can be pre-trained and fine-tuned for different downstream tasks by just changing the head, as we will see later. You can check the encoders provided by [AllenNLP](https://github.com/allenai/allennlp/tree/master/allennlp/modules/seq2seq_encoders) or even write your own by implementing the [Seq2SeqEncoder interface].

For defining encoders, biome.text builds on top of the `Seq2SeqEncoder` abstraction from AllenNLP, which brings many configuration possibilities, that go from RNNs to the official PyTorch Transformer implementation.
### Head
The other key component for supporting flexible transfer learning are task heads. A head defines the NLP task (e.g., text classification, token-level classification, language modelling) and specific features related to the task, for example the labels of the a text classifier (``positive`` and ``negative`` in our example). You can check available heads in the [API documentation](../api/biome/text/modules/heads/).


## Datasource
Data sources provide an easy way to load data for training, evaluation and inference coming from different sources: [csv, parquet, json, or Excel spreadsheets among others](../api/biome/text/data/readers.md#biome-text-data-readers). 

Data sources map data into a lazy [Dask DataFrame](https://docs.dask.org/en/latest/dataframe.html), so you can easily inspect them and manipulate them using familiar Pandas DataFrame operations.

Coming back to our example:

``` python{9}
from biome.text import Pipeline, VocabularyConfiguration
from biome.text.data import DataSource

pipeline = Pipeline.from_config({
    "name": "my-first-classifier", 
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = DataSource(source='training_data.csv')
pipeline.create_vocabulary(VocabularyConfiguration(sources=[train_ds]))

pipeline.train(
    output="path_to_store_training_run_output",
    training=train_ds
)
```
Here we instantiate a ``DataSource`` from a csv file that looks like this:

| text        | label           | 
| ------------- |:-------------:| 
| I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...     | positive |
| Phil the Alien is one of those quirky films where the humour is based around the oddness of ...     | negative      |  
| ... | positive      | 

Data sources can also be created from [YAML configuration files](../api/biome/text/data/datasource.md#from-yaml), which might be handy for automating training and evaluation pipelines.

Fields in data sources are intimately related to what the pipeline expects as input and output features. In our example, we are defining a text classification model which expects a ``text`` and a ``label``. In cases where users don't have the option to align the fields of the data with the features of the model, the class ``DataSource`` provides a `mapping`` parameter. Imagine our data set looked like this:

| title     | review        | label         | 
|-----------| ------------- |:-------------:| 
|    Cool summer movie!      | I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...     | positive |
|          Horrible horror movie | Phil the Alien is one of those quirky films where the humour is based around the oddness of ...     | negative      |  
|           | ...            | positive      | 

Using the ``mapping`` functionality we could not only work with this data set schema by setting a mapping: 

```python
train_ds = DataSource(source='training_data.csv', mapping={'text': 'review'}) 
```

but we could also combine both *title* and *review* to feed them as input features like so: 

```python
train_ds = DataSource(
    source='training_data.csv', 
    mapping={'text': ['title', 'review']}
) 
```
biome.text was created with semi-structured data problems in mind, so it provides specialized models for learning from structured records such as the [RecordClassification](../api/biome/text/modules/heads/classification/record_classification.md#recordclassification) head, which lets you define mappings to arbitrary input fields and combine their vector representations in a hierarchical way (e.g., combining encoders at field and record level)

You can find more info about data sources and mappings in the [API documents](../api/biome/text/datasource.md#datasource).
## Vocabulary
``` python{10}
from biome.text import Pipeline, VocabularyConfiguration
from biome.text.data import DataSource

pipeline = Pipeline.from_config({
    "name": "my-first-classifier", 
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = DataSource(source='training_data.csv')
pipeline.create_vocabulary(VocabularyConfiguration(sources=[train_ds]))

pipeline.train(
    output="path_to_store_training_run_output",
    training=train_ds
)
```
