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
For now biome.text allows you to extract 3 features from your input: `word`, `char` and `transforrmers`.
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

In this example we show a more complex configuration, mainly due to the chosen [*RecordPairClassification*](../../api/biome/text/modules/heads/classification/record_pair_classification.html#recordpairclassification) head.

We will start with the `tokenizer` part, where, apart from tokenization parameters, you can also specify some `text_cleaning` rules that will be applied to the input before the tokenization happens.

As input feature we choose the [`char` feature](../../api/biome/text/features.md#charfeatures) that uses character embeddings and a `Seq2VecEncoder` to encode the tokens.
Here we choose a *gru* encoder and apply a little dropout afterwards.

The ...

### TokenClassification with transformers and char features

...

## Trainer

## Vocabulary
