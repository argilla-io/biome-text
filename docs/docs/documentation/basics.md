# The basics
The library is built around a few simple concepts. This section explains everything you need to know to get started (feel free to jump into the sections you are more interested in):

[[toc]]

Before going into details, let's see a simple example:
``` python
from biome.text import Pipeline, Dataset, Trainer

pipeline = Pipeline.from_config({
    "name": "my-first-classifier",
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = Dataset.from_csv('training_data.csv')

trainer = Trainer(
    pipeline=pipeline,
    train_dataset=train_ds
)
trainer.fit()
```

The above example trains a text classifier from scratch by configuring a ``Pipeline``, making a ``Dataset`` from a csv file and passing both to a `Trainer`.
Let's dive into the details.

## Pipeline

Pipelines are the main entry point to the library. A ``Pipeline`` bundles components needed to train, evaluate and use custom NLP models.

Pipelines encompass tokenization, feature processing, model configuration and actions such as serving or inference.

Let's continue with our example:

``` python{3-6}
from biome.text import Pipeline, Dataset, Trainer

pipeline = Pipeline.from_config({
    "name": "my-first-classifier",
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = Dataset.from_csv('training_data.csv')

trainer = Trainer(
    pipeline=pipeline,
    train_dataset=train_ds
)
trainer.fit()
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
In this example, we only define the `name` and the task we want to train our model on, using the ``head`` parameter, the rest is configured from defaults.
In *biome.text* we try to provide sensible defaults so you don't have to configure everything just to start experimenting, but there are many things you can tune and configure.

In particular, a `Pipeline` has the following configurable components:

### Tokenizer
The tokenizer defines how we want to process the text of our input features. Tokenizers are based on [spaCy tokenizers](https://spacy.io/api/tokenizer) and have the following main configuration options:

1. ``lang``: the main language of the text to be tokenized (default is English). Here you can use available [spaCy model codes](https://spacy.io/usage/models/).
2. ``segment_sentences``: enable sentence splitting for text within your input features, which is especially relevant for long text classification problems.
3. ``text_cleaning``: simple python functions to pre-process text before tokenization. You can define your own but *biome.text* provides pre-defined functions for things like cleaning up html tags or remove extra blank spaces.

### Features
Features are a central concept of the library. Building on the flexibility of AllenNLP, *biome.text* gives you the ability of combining [Word](../api/biome/text/features.md#wordfeatures), [Character](../api/biome/text/features.md#charfeatures) and other input features easily. There are many things which can be configured here: the size of the embeddings, encoder type (e.g., CNNs or RNNs) for character encoding, pre-trained word vectors, and other things.

To learn more about how to configure and use Features, see the [FeaturesConfiguration API docs](../api/biome/text/configuration.md#featuresconfiguration).

### Encoder
To support transfer learning, models are structured into a model "backbone" for processing and encoding features and a "task" head for a certain NLP task.

The ``Encoder`` is a central piece of the backbone. It's basically a sequence to sequence or seq2seq encoder, which "contextualizes" textual features in the context of a task (supervised or unsupervised). In this way, the encoder can be pre-trained and fine-tuned for different downstream tasks by just changing the head, as we will see later. You can check the encoders provided by [AllenNLP](https://github.com/allenai/allennlp/tree/master/allennlp/modules/seq2seq_encoders) or even write your own by implementing the [Seq2SeqEncoder interface](https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2seq_encoders/seq2seq_encoder.py).

For defining encoders, *biome.text* builds on top of the `Seq2SeqEncoder` abstraction from AllenNLP, which brings many configuration possibilities, that go from RNNs to the official PyTorch Transformer implementation.

### Head
Task heads are the other key component to support flexible transfer learning. A head defines the NLP task (e.g., text classification, token-level classification, language modelling) and specific features related to the task, for example the labels of a text classifier (``positive`` and ``negative`` in our example).

You can check available heads in the [API documentation](../api/biome/text/modules/heads/).


## Dataset
The `Dataset` class provides an easy way to load data for training, evaluation and inference coming from different sources: [csv, json or from pandas DataFrames among others](../api/biome/text/dataset.md#dataset).

It is a very thin wrapper around HuggingFace's awesome [datasets.Dataset](https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset).
Most of HuggingFace's `Dataset` API is exposed, and you can check out their nice [documentation](https://huggingface.co/docs/datasets/master/processing.html) on how to work with data in a `Dataset`.

Coming back to our example:

``` python{8}
from biome.text import Pipeline, Dataset, Trainer

pipeline = Pipeline.from_config({
    "name": "my-first-classifier",
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = Dataset.from_csv('training_data.csv')

trainer = Trainer(
    pipeline=pipeline,
    train_dataset=train_ds
)
trainer.fit()
```
Here we instantiate a ``Dataset`` from a csv file that looks like this:

| text        | label           |
| ------------- |:-------------:|
| I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...     | positive |
| Phil the Alien is one of those quirky films where the humour is based around the oddness of ...     | negative      |
| ... | positive      |

Columns in data sets are intimately related to what the pipeline expects as input and output features. In our example, we are defining a text classification model which expects a ``text`` and a ``label`` column.
In cases where users don't have the option to align the columns of the data with the features of the model, the ``Dataset`` class provides a `rename_column_()` and a `map()` method. Imagine our data set looked like this:

| title     | review        | label         |
|-----------| ------------- |:-------------:|
|    Cool summer movie!      | I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...     | positive |
|          Horrible horror movie | Phil the Alien is one of those quirky films where the humour is based around the oddness of ...     | negative      |
|           | ...            | positive      |

Using the ``rename_column_()`` method we could not only work with this data set schema by renaming one column:

```python
train_ds.rename_column_('review', 'text')
```

but we could also combine both *title* and *review* to feed them as input features using the `map()` method:

```python
train_ds = train_ds.map(lambda row: {"text": row["titile"] + row["review"]})
```
*biome.text* was created with semi-structured data problems in mind, so it provides specialized models for learning from structured records such as the [RecordClassification](../api/biome/text/modules/heads/classification/record_classification.md#recordclassification) head, which lets you define mappings to arbitrary input fields and combine their vector representations in a hierarchical way (e.g., combining encoders at field and record level)

## Vocabulary
For doing NLP with neural networks, your NLP pipeline needs to turn words, subwords and/or characters into numbers. A typical process consists of tokenizing the text, and mapping word (or sub-word) tokens and maybe characters into integers or indexes. This process is often referred to as "indexing".

In order to support this you need a ``Vocabulary``, which holds a mapping from tokens and characters to numerical ids (if you are interested, this post provides a good [overview](https://blog.floydhub.com/tokenization-nlp/)). This mapping can be learned and built during pre-training phases as is the case for word piece models and generally sub-word models like those provided by [Huggingface tokenizers](https://github.com/huggingface/tokenizers)).

A more classical approach is to build or extend an existing vocabulary from training and validation data sets. For certain use cases, in highly specialized domains, this is sometimes the best way to proceed.

*biome.text* takes care of building your vocabulary automatically if necessary.
By default, it will build the vocab based on the training data set, but if you want more control over this step you can pass a `VocabularyConfiguration` instance to the `Trainer` method:

```python
from biome.text import VocabularyConfiguration

vocab_config = VocabularyConfiguration(include_valid_data=True, max_vocab_size=1000)
trainer = Trainer(
    pipeline=pipeline,
    train_dataset=train_ds,
    vocab_config=vocab_config,
)
```

Here we create our vocabulary using the training and validation data set and limit it to 1000 entries.
AllenNLP provides neat abstractions for dealing with multi-feature vocabularies (e.g., chars, words, etc.) and *biome.text* builds on top of those abstractions to make it easy to create, reuse and extend vocabularies.

To learn more about how to configure and use the `Vocabulary`, see the [VocabularyConfiguration API docs](../api/biome/text/configuration.md#vocabularyconfiguration)

## Train
Once we have everything ready, we can use the [`Trainer`](../api/biome/text/trainer.md) to train our pipeline.
Our `Trainer` is a light wrapper around the amazing [`Pytorch Lightning` trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).
Going back to our example:
``` python{10-14}
from biome.text import Pipeline, Dataset, Trainer

pipeline = Pipeline.from_config({
    "name": "my-first-classifier",
    "head": {"type": "TextClassification", "labels": ["positive", "negative"]}
})

train_ds = Dataset.from_csv('training_data.csv')

trainer = Trainer(
    pipeline=pipeline,
    train_dataset=train_ds
)
trainer.fit()
```
By default, the `trainer.fit()` method will create an `output` folder with a `model.tar.gz` and a `metrics.json` file:
- `model.tar.gz`: This is the more relevant file since it contains the trained model weights, the vocabulary and the pipeline configuration, which is everything we need to explore, serve or fine-tune the trained model.
- `metrics.json`: It merely contains a summary of the logged metrics during the training and can be useful if you want to quickly compare different output folders.

The `Pytorch Lightning` trainer under the hood will also create a `training_logs` folder in your working directory that contains all your checkpoints and logged metrics, which you can visualize with [tensorboard](https://www.tensorflow.org/tensorboard/) for example.

In the example, we only provide a `train_dataset` which is of course not recommended for most use cases, where you need at least a validation set and desirably a test set.
We also do not set anything related to the training process, such as optimizer, learning rate, epochs and so on.
The library provides basic defaults for this just to get started.
When further experimenting, you will probably need to use a [TrainerConfiguration](../api/biome/text/configuration.md#trainerconfiguration) object:

```python
from biome.text import TrainerConfiguration

trainer_config = TrainerConfiguration(optimizer={"type": "adamw", "lr": 0.01})
trainer = Trainer(
    pipeline=pipeline,
    train_dataset=train_ds,
    trainer_config=trainer_config,
)
```

## Using pre-trained pipelines

As mentioned above, ``model.tar.gz`` files resulting from a training run contain everything you need to start using your trained pipeline.

Let's see how:

```python
pipeline = Pipeline.from_pretrained('output/model.tar.gz')
```
After loading the pipeline, you can use it for predictions:

```python
pipeline.predict(text='Good movie indeed!')
```

### Predict

The arguments for the [Pipeline.predict()](../api/biome/text/pipeline.md#predict) method will be aligned to match our model input features (`text` in this case).
Models such as the `RecordClassifier` can be used to define more fine-grained input features, such as, for example, an email classifier with two fields `subject` and `body`:
```python
pipeline.predict(subject='Hi!', body='Hi, hope you are well..')
```

Some of our heads support attributing the input to the prediction by means of integrated gradients.
Following will add an `attributions` key in the prediction dictionary:

```python
pipeline.predict(subject='Hi!', body='Hi, hope you are well..', add_attributions=True)
```

### Training and transfer learning

After loading a pipeline, you can continue the training for the same task with new data:
```python
trainer = Trainer(
    pipeline=loaded_pipeline,
    train_dataset=new_training_ds
)
trainer.fit(output_dir="finetuned_output")
```

Another thing you can do is to use this pre-trained pipeline for related task, for example another classifier with different labels but a similar domain:
```python{3-6}
from biome.text.modules.heads import TextClassification

loaded_pipeline.set_head(
    TextClassification,
    labels = ["horror_movie", "comedy", "drama", "social"]
)

trainer = Trainer(
    pipeline=loaded_pipeline,
    train_dataset=new_training_ds
)
trainer.fit(output_dir="finetuned_output")
```
Here we just use the [Pipeline.set_head()](../api/biome/text/pipeline.md#set-head) method to set a new task head which classifies film categories instead of review sentiment.

The more common "pre-training a language model + fine-tuning on downstream tasks" is also supported by using a [LanguageModelling](../api/biome/text/modules/heads/language_modelling.md#languagemodelling) head for pre-training.

### Serve
*biome.text* uses the awesome [FastAPI](https://fastapi.tiangolo.com/) library to give you a simple REST endpoint with the [biome serve](../api/biome/text/cli/serve.md) CLI command, which provides methods aligned with your input features (e.g., a method accepting `subject` and `body` parameters).
In a terminal just type in:
```terminal
biome serve path/to/your/model.tar.gz
```



## Next steps

The best place to start is the tutorials section.

You can also find detailed [API docs](../api/) and use the search bar on top to find out more about specific concepts and features.
