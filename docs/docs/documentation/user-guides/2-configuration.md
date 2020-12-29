# Configurations

This section provides some example configurations for common use cases.

In general, we try to configure all our user exposed elements with `Configuration` classes whose parameters you can look up in the [configuration API](../../api/biome/text/configuration.md).

<h2>Table of contents</h2>

[[toc]]

## Pipeline

The pipeline is the [main entry point](../basics.md#pipeline) to the library and is therefore highly configurable.
Its configuration is a composition of many lower-level `Configuration` classes, each one taking care of configuring a specific task in the pipeline.
For a list of the main pipeline components see the [introduction to the `Pipeline` class](../basics.md#pipeline).

Let us have a more detailed look at the pipeline configuration with the help of examples.

::: tip Tip

Since the input dimensions of the single model components depend on the previous component, we automatically compute them for you.
Hence, you just have to worry about the embedding dimension of your features and hidden dimensions of your model components.

:::

### Text classification with word features

```python
pipeline_config = {
    "name": "text_classification",
    "features": {
        "word": {
            "embedding_dim": 300,
            "lowercase_tokens": True,
        },
    },
    "head": {
        "type": "TextClassification",
        "labels": ["positive", "negative"],
        "pooler": {
            "type": "gru",
            "num_layers": 1,
            "hidden_size": 32,
            "bidirectional": True,
        },
    },
}
```

The first configuration key should always be the name of your pipeline. Choose something descriptive and short.

We then tell the pipeline to extract only the [word feature](../../api/biome/text/features.md#wordfeatures) from our input.
For now *biome.text* allows you to extract 3 features from your input: `word`, `char` and `transforrmers`.
You can use only one or combine all three of them.

The type of the head is provided in the `type` key. In this case we will use a [*TextClassification*](../../api/biome/text/modules/heads/classification/text_classification.md) head and provide the labels it should predict in the `labels` key.
For a list of available `head` types, check out the [NLP tasks](1-nlp-tasks.md) section.
The `pooler` of our *TextClassification* head must be one of AllenNLP's [`Seq2VecEncoder`](https://docs.allennlp.org/master/api/modules/seq2vec_encoders/seq2vec_encoder/).
This abstract class encompasses their custom modules and native Pytorch modules, that take as input a sequence of vectors and output a single vector.
Here we choose the *gru* type whose parameters you can look up in their extensive [documentation](https://docs.allennlp.org/master/api/modules/seq2vec_encoders/pytorch_seq2vec_wrapper/#gruseq2vecencoder).

### RecordPairClassification with char features

```python
pipeline_config = {
    "name": "my_record_pair_classifier",
    "tokenizer": {"text_cleaning": {"rules": ["strip_spaces"]}},
    "features": {
        "char": {
            "embedding_dim": 64,
            "lowercase_characters": True,
            "encoder": {
                "type": "gru",
                "hidden_size": 128,
                "num_layers": 1,
                "bidirectional": True,
            },
            "dropout": 0.1,
        },
    },
    "head": {
        "type": "RecordPairClassification",
        "labels": ["duplicate", "not_duplicate"],
        "dropout": 0.1,
        "field_encoder": {
            "type": "gru",
            "bidirectional": False,
            "hidden_size": 128,
            "num_layers": 1,
        },
        "record_encoder": {
            "type": "gru",
            "bidirectional": True,
            "hidden_size": 64,
            "num_layers": 1,
        },
        "matcher_forward": {
            "is_forward": True,
            "num_perspectives": 10,
            "with_full_match": False,
        },
        "matcher_backward": {
            "is_forward": False,
            "num_perspectives": 10,
        },
        "aggregator": {
            "type": "gru",
            "bidirectional": True,
            "hidden_size": 64,
            "num_layers": 1,
        },
        "classifier_feedforward": {
            "num_layers": 1,
            "hidden_dims": [32],
            "activations": ["relu"],
            "dropout": [0.1],
        },
        "initializer": {
            "regexes": [
                ["_output_layer.weight", {"type": "xavier_normal"}],
                ["_output_layer.bias", {"type": "constant", "val": 0}],
                [".*linear_layers.*weight", {"type": "xavier_normal"}],
                [".*linear_layers.*bias", {"type": "constant", "val": 0}],
                [".*weight_ih.*", {"type": "xavier_normal"}],
                [".*weight_hh.*", {"type": "orthogonal"}],
                [".*bias.*", {"type": "constant", "val": 0}],
                [".*matcher.*match_weights.*", {"type": "kaiming_normal"}],
            ]
        },
    },
}
```

In this example we show a more complex configuration, mainly due to the chosen *RecordPairClassification* head.

We will start with the `tokenizer` part, where, apart from tokenization parameters, you can also specify some `text_cleaning` rules that will be applied to the input before the tokenization happens.

As input feature we choose the [`char` feature](../../api/biome/text/features.md#charfeatures) that uses character embeddings and a `Seq2VecEncoder` to encode the tokens.
Here we choose a *gru* encoder and apply a little dropout afterwards.

The *RecordPairClassification* head has a lot of components that are described in detail in [the API docs](../../api/biome/text/modules/heads/classification/record_pair_classification.html#recordpairclassification) and the references therein.
As in the *TextClassification* head above, in the `labels` key we have to provide the labels we want our classifier to predict.
We then configure all the different layers of our head (various `Seq2VecEncoder`, `Seq2SeqEncoder`, `BiMpmMatching` and a `FeedForward`) and also get the possibility of setting [initializers](https://docs.allennlp.org/main/api/nn/initializers/) for all the model parameters.
For this we first have to write down a regex expression of the parameter group name and then specify the initializer.
You can lookup the available parameter groups with the `Pipeline.named_trainable_parameters` property, once the pipeline is created.


### TokenClassification with transformers and char features

```python
pipeline_config = {
    "name": "my_ner_pipeline",
    "features": {
        "transformers": {
            "model_name": "sshleifer/tiny-distilbert-base-cased",
            "trainable": True,
            "max_length": 512,
        },
        "char": {
            "embedding_dim": 32,
            "encoder": {
                "type": "gru",
                "num_layers": 1,
                "hidden_size": 64,
                "bidirectional": True,
            },
            "dropout": 0.1,
        },
    },
    "encoder": {
        "type": "lstm",
        "num_layers": 1,
        "hidden_size": 256,
        "bidirectional": True,
    },
    "head": {
        "type": "TokenClassification",
        "labels": ["PERSON", "LOCATION", "DATE", "ORGANIZATION"],
        "label_encoding": "BIOUL",
        "feedforward": {
            "num_layers": 1,
            "hidden_dims": [256],
            "activations": ["relu"],
            "dropout": [0.1],
        },
    },
}
```

In this example we combine two input features:
- [`transformers`](../../api/biome/text/features.md#transformersfeatures): this feature embeds word or word-piece tokens with a pretrained transformer from [HuggingFace's model hub](https://huggingface.co/). See the [transformers tutorial](../tutorials/4-Using_Transformers_in_biome_text.md) for details;
- [`char`](../../api/biome/text/features.md#charfeatures): this feature uses character embeddings and a `Seq2VecEncoder` to encode the tokens. See the example [above](./2-configuration.md#RecordPairClassification-with-char-features).

The resulting word vectors of both features are then simply concatenated and passed on to the next pipeline component.

While the word vectors of the transformers feature are already contextualized, the char feature vectors are not.
Therefore, we add an `encoder` layer to the pipeline that consists of a [`Seq2SeqEncoder`](https://docs.allennlp.org/main/api/modules/seq2seq_encoders/seq2seq_encoder/) and contextualizes the word vectors by means of, in our case, a bidirectional *lstm*.

In the [`TokenClassification` head](../../api/biome/text/modules/heads/token_classification.md) we need to provide the labels and the [encoding scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) of our labels (tags), in our case *BIOUL*.
We further specify a [feedforward layer](https://docs.allennlp.org/main/api/modules/feedforward/) that is applied to each of the output vector of the encoder sequence.

The `TokenClassification` head accepts as training input pretokenized text together with its tags, or raw text together with entities (see the [API docs](../../api/biome/text/modules/heads/token_classification.md#featurize) for details).

## Trainer

The training in *biome.text* uses AllenNLP's feature rich [`GradientDescentTrainer`](https://docs.allennlp.org/main/api/training/trainer/#gradientdescenttrainer) and is configured via the [`TrainerConfiguration`](../../api/biome/text/configuration.md#trainerconfiguration) class.
The configuration is directly consumed by the `Pipeline.train` method and covers options for most common use cases.

### AdamW optimizer with linear warm-up and decay

```python
from biome.text import TrainerConfiguration

trainer_config = TrainerConfiguration(
    optimizer={
        "type": "adamw",
        "lr": 0.001,
    },
    learning_rate_scheduler={
        "type": "linear_with_warmup",
        "num_epochs": 5,
        "num_steps_per_epoch": 1000,
        "warmup_steps": 100,
    },
    num_epochs=5,
    batch_size=8,
    patience=None,
)
```

In this example we are using a [AdamW optimizer](https://docs.allennlp.org/main/api/training/optimizers/) with a learning rate of 0.001.
We combine this optimizer with a [learning rate scheduler](https://docs.allennlp.org/main/api/training/learning_rate_schedulers/learning_rate_scheduler/) that linearly increases the learning rate from 0 to the learning rate specified in the optimizer over 100 steps (*warmup_steps*), and then linearly decreases the learning rate again to 0 until the end of the training.
In this case the number of epochs in the learning rate scheduler, and the number of epochs in the `TrainerConfiguration` should be the same.
The `num_steps_per_epoch` parameter should reflect the number of training examples divided by the `batch_size` of the training.

By default, the `patience` of the trainer is set to 2 epochs.
However, when providing a learning rate scheduler it is advised to set the `patience` to *None* in order not to interfere with the schedule.

## Vocabulary

The creation of the vocabulary is managed automatically by *biome.text* and for most use cases the default settings are perfectly fine.
If you want to do something more specific with your vocabulary you can provide a [`VocabularyConfiguration`](../../api/biome/text/configuration.md#vocabularyconfiguration) object to the `Pipeline.train()` method.

### Limit vocab to pretrained word vectors

```python
from biome.text import Dataset, VocabularyConfiguration
train_ds = Dataset.from_dict({"text": ["test text"], "label": ["test_label"]})

vocab_config = VocabularyConfiguration(
    datasets=[train_ds],
    pretrained_files={"word": "path/to/pretrained_words.txt"},
    only_include_pretrained_words=True,
)
```

In this example we limit the vocabulary to words that are listed in a file with pretrained word vectors.

A `VocabularyConfiguration` must always include a list of `Dataset`s from which the vocabulary is created.
By default, this list only includes the training data provided to the `Pipeline.train()` method.

The `pretrained_files` parameter is a mapping in form of a dictionary, in which the key indicates the namespace of the feature for which the pretrained file is valid.
This parameter enables a few other parameters, such as the `only_include_pretrained_words` that allows you to only include words in the vocabulary that are present in your data **and** in the pretrained files.
