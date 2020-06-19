# The basics
The library is built around a few simple concepts. This section explains everything you need to know to get started (feel free to jump into the sections you are more interested in):

[[toc]]

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

training_results = pipeline.train(
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

training_results = pipeline.train(
    output="path_to_store_training_run_output",
    training=train_ds
)
```
Here we configure a ``Pipeline`` from scratch using a dictionary. A pipeline can be created in two ways:

1. Using the ``Pipeline.from_config()`` method, which accepts a dict or a [PipelineConfiguration](../api/biome/text/configuration.md#pipelineconfiguration)

2. Using YAML configuration files with the `Pipeline.from_yaml()`. The YAML config file in our example would be following:

```yaml
name: my-first-classifier
head:
    type: TextClassification
    labels: ["positive", "negative"]
```
In this example, we only define the `name` and the task we want to train our model on, using the ``head`` parameter, the rest is configured from defaults. In biome.text we try to provide sensible defaults so you don't have to configure everything just to start experimenting, but there are many things you can tune and configure.

In particular, a `Pipeline` has the following configurable components:

### Tokenizer
The tokenizer defines how we want to process the text of our input features. Tokenizers are based on [spaCy tokenizers](https://spacy.io/api/tokenizer) and have the following main configuration options:

1. ``lang``: the main language of the text to be tokenized (default is English). Here you can use available [spaCy model codes](https://spacy.io/usage/models/).
2. ``segment_sentences``: enable sentence splitting for text within your input features, which is especially relevant for long text classification problems.
3. ``text_cleaning``: simple python functions to pre-process text before tokenization. You can define your own but biome.texts provides pre-defined functions for things like cleaning up html tags or remove extra blank spaces.

### Features
Features are a central concept of the library. Building on the flexibility of AllenNLP, biome.text gives you the ability of combining [Word](../api/biome/text/features.md#wordfeatures) and [Character](../api/biome/text/features.md#charfeatures) features easily. There are many things which can be configured here: the size of the embeddings, encoder type (e.g., CNNs or RNNs) for character encoding, pre-trained word vectors, and other things.

To learn more about how to configure and use Features, see the [FeaturesConfiguration API docs](../api/biome/text/configuration.md#featuresconfiguration).

### Encoder
To support transfer learning, models are structured into a model "backbone" for processing and encoding features and a "task" head for a certain NLP task.

The ``Encoder`` is a central piece of the backbone. It's basically a sequence to sequence or seq2seq encoder, which "contextualizes" textual features in the context of a task (supervised or unsupervised). In this way, the encoder can be pre-trained and fine-tuned for different downstream tasks by just changing the head, as we will see later. You can check the encoders provided by [AllenNLP](https://github.com/allenai/allennlp/tree/master/allennlp/modules/seq2seq_encoders) or even write your own by implementing the [Seq2SeqEncoder interface](https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2seq_encoders/seq2seq_encoder.py).

For defining encoders, biome.text builds on top of the `Seq2SeqEncoder` abstraction from AllenNLP, which brings many configuration possibilities, that go from RNNs to the official PyTorch Transformer implementation.

### Head
Task heads are the other key component to support flexible transfer learning. A head defines the NLP task (e.g., text classification, token-level classification, language modelling) and specific features related to the task, for example the labels of a text classifier (``positive`` and ``negative`` in our example).

You can check available heads in the [API documentation](../api/biome/text/modules/heads/).


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

training_results = pipeline.train(
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

Columns in data sources are intimately related to what the pipeline expects as input and output features. In our example, we are defining a text classification model which expects a ``text`` and a ``label`` column. In cases where users don't have the option to align the columns of the data with the features of the model, the ``DataSource`` class provides a `mapping` parameter. Imagine our data set looked like this:

| title     | review        | label         |
|-----------| ------------- |:-------------:|
|    Cool summer movie!      | I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...     | positive |
|          Horrible horror movie | Phil the Alien is one of those quirky films where the humour is based around the oddness of ...     | negative      |
|           | ...            | positive      |

Using the ``mapping`` functionality we could not only work with this data set schema by setting a mapping:

```python{3}
train_ds = DataSource(
    source='training_data.csv',
    mapping={'text': 'review'}
)
```

but we could also combine both *title* and *review* to feed them as input features:

```python{3}
train_ds = DataSource(
    source='training_data.csv',
    mapping={'text': ['title', 'review']}
)
```
biome.text was created with semi-structured data problems in mind, so it provides specialized models for learning from structured records such as the [RecordClassification](../api/biome/text/modules/heads/classification/record_classification.md#recordclassification) head, which lets you define mappings to arbitrary input fields and combine their vector representations in a hierarchical way (e.g., combining encoders at field and record level)

You can find more info about data sources and mappings in the [API documents](../api/biome/text/datasource.md#datasource).

## Vocabulary
For doing NLP with neural networks, your NLP pipeline needs to turn words, subwords and/or characters into numbers. A typical process consists of tokenizing the text, and mapping word (or sub-word) tokens and maybe characters into integers or indexes. This process is often referred to as "indexing".

In order to support this you need a ``Vocabulary``, which holds a mapping from tokens and characters to numerical ids (if you are interested this post provides a good [overview](https://blog.floydhub.com/tokenization-nlp/. This mapping can be learned and built during pre-training phases as is the case for word piece models and generally sub-word models like those provided by [Huggingface tokenizers](https://github.com/huggingface/tokenizers)).

A more classical approach is to build or extend an existing vocabulary from training and validation data sets. For certain use cases, in highly specialized domains, this is sometimes the best way to proceed.

Coming back to our example:

``` python{10}
from biome.text import Pipeline, VocabularyConfiguration
from biome.text.data import DataSource

pipeline = Pipeline.from_config({
    "name": "my-first-classifier",
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = DataSource(source='training_data.csv')
pipeline.create_vocabulary(VocabularyConfiguration(sources=[train_ds]))

training_results = pipeline.train(
    output="path_to_store_training_run_output",
    training=train_ds
)
```
Here we create our vocabulary from scratch using the training data source. AllenNLP provides neat abstractions for dealing with multi-feature vocabularies (e.g., chars, words, etc.) and biome.text builds on top of those abstractions to make it easy to create, reuse and extend vocabularies.

To learn more about `Vocabulary` configuration options and usage, see the [VocabularyConfiguration API docs](../api/biome/text/configuration.md#vocabularyconfiguration)

## Train
Once we have everything ready, we can use Pipeline for training our model using the [train](../api/biome/text/pipeline.md#pipeline). Going back to our example:
``` python{13}
from biome.text import Pipeline, VocabularyConfiguration
from biome.text.data import DataSource

pipeline = Pipeline.from_config({
    "name": "my-first-classifier",
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = DataSource(source='training_data.csv')
pipeline.create_vocabulary(VocabularyConfiguration(sources=[train_ds]))

training_results = pipeline.train(
    output="path_to_store_training_run_output",
    training=train_ds
)
```
Here the training output will be saved in a folder. It will contain the trained model weights and the metrics, as well as the vocabulary and a log folder for visualizing the training metrics with [tensorboard](https://www.tensorflow.org/tensorboard/).

The most relevant file in this folder will be the ``model.tar.gz`` file, which bundles everything we need for loading the trained model for exploration, serving or fine-tuning.

In the example, we only provide a ``training_ds`` which is of course not recommended for most use cases, where you need at least a validation set and desirably a test set. We also do not set anything related to the training process, such as optimizer, learning rate, epochs and so on. The library provides basic defaults for this just to get started. When further experimenting, you will probably need to use a trainer, configured with a [TrainerConfiguration](../api/biome/text/configuration.md#trainerconfiguration) object.

## Using pre-trained pipelines

As mentioned above, ``model.tar.gz`` files resulting from a training run contain everything you need to start using your trained pipeline.

Let's see how:

```python
pipeline = Pipeline.from_pretrained('path_to_store_training_run_output/model.tar.gz')
```
After loading the pipeline, you can use it for predictions:

```python
pipeline.predict(text='Good movie indeed!')
```

### Predict

The [predict](../api/biome/text/pipeline.md#predict) method for our pipeline will be aligned to match our model input features (`text` in this case). Models such as the `RecordClassifier` can be used to define more fine-grained input features, such as for example an email classifier with two fields `subject` and `body`:
```python
pipeline.predict(subject='Hi!', body='Hi, hope you are well..')
```

### Explain
Another thing we can do with a trained pipeline is to use the [explain](../api/biome/text/pipeline.md#explain) method to get the attribution of each token by means of [integrated gradients](https://arxiv.org/abs/1703.01365) for those heads that support it:
```python
pipeline.explain(text='Good movie indeed!')
```

### Serve
Pipeline use the awesome [FastAPI](https://fastapi.tiangolo.com/) library to give you simple REST endpoint with the [serve](../api/biome/text/pipeline.md#serve) method, which provides methods aligned with your input features (e.g., a method accepting `subject` and `body` parameters):

```python
pipeline.serve(port=9090)
```

### Explore
In order to support users with fine-grained error analysis and empower them to improve their models with informed decisions, the [explore](../api/biome/text/pipeline.md#explore) method launches a UI (inside your notebook if you are using Jupyter).

```python
pipeline.explore(
    train_ds,
    explain=True
)
```

This search-based UI can help:

 - to quickly identify difficult examples in the training, validation, test or any other data set.
 - to inform users about corner cases and which labels need more examples.
 - to find the values for confidence thresholds to use the model in production.
 - to help domain experts understand the strengths and limitations of the model.
 - in summary, get a sense of the model interacting with data.

:::tip
For the UI to work you need a running [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) instance. We recommend installing [Elasticsearch with docker](https://www.elastic.co/guide/en/elasticsearch/reference/7.7/docker.html#docker-cli-run-dev-mode).
:::

### Training and transfer learning

After loading a pipeline, you can keep on training it for the same task with new data:
```python
training_results = pipeline.train(
    output="path_to_store_further_training",
    training=new_training_ds
)
```
:::tip Tip
For further training, you can use the [extend_vocab](../api/biome/text/pipeline.md#pipeline) parameter to extend the vocabulary with the new data sets.
:::

Another thing you can do is to use this pre-trained pipeline for related task, for example another classifier with different labels but a similar domain:
```python{3-6}
from biome.text.modules.heads import TextClassification

pipeline.set_head(
    TextClassification,
    labels = ["horror_movie", "comedy", "drama", "social"]
)

training_results = pipeline.train(
    output="path_to_fine_tuned",
    training=categories_training_ds
)
```
Here we just use the [Pipeline.set_head()](../api/biome/text/pipeline.md#set-head) method to set a new task head which classifies film categories instead of review sentiment.

The more common "pre-training a language model + fine-tuning on downstream tasks" is also supported by using a [LanguageModelling](../api/biome/text/modules/heads/language_modelling.md#languagemodelling) head for pre-training.


## Next steps

The best place to start is the tutorials section.

Also you find detailed [API docs](../api/) and use the search feature to find out more about specific concepts and features.
