# Configuration

<h2>Table of contents</h2>

[[toc]]

This section describes the possible options you can specify in
configuration files for biome text. For now we just point to the
responsible parts in the code.

# Pipeline configuration

```python
from biome.text import Pipeline

pl = Pipeline.from_yaml("path/to/pipeline.yml")
```

The content of the yaml file is embedded in an `allennlp.common.Params`
and passed on to
```python
from biome.text.configuration import PipelineConfiguration
class PipelineConfiguration(FromParams):
    def __init__(
        self,
        name: str,
        features: FeaturesConfiguration,
        head: TaskHeadSpec,
        tokenizer: Optional[TokenizerConfiguration] = None,
        encoder: Optional[Encoder] = None,
    )
```
via the `PipelineConfiguration.from_params()` method. This means the init
arguments get instantiated (see `allennlp.common.FromParams`).

The `PipelineConfiguration.__init__()` gives us the list of possible
first level sections in the `pipeline.yml`:
 - name
 - features
 - head
 - tokenizer
 - encoder

## features

This section of the `pipeline.yml` file is passed on to the
```python
class biome.text.api_new.configuration.FeaturesConfiguration(FromParams):
    """Features configuration spec"""

    def __init__(
        self,
        words: Optional[Dict[str, Any]] = None,
        chars: Optional[Dict[str, Any]] = None,
        **extra_params
    )
```

**All** `__init__` arguments are converted to attributes and are passed on to
`biome.text.featurizer.InputFeaturizer` in the `FeaturesConfiguration.compile()` method.

```python
class InputFeaturizer:
    __DEFAULT_CONFIG = {"embedding_dim": 8, "lowercase_tokens": True}
    __INDEXER_KEYNAME = "indexer"
    __EMBEDDER_KEYNAME = "embedder"

    WORDS = _WordFeaturesSpecs.namespace
    CHARS = _CharacterFeaturesSpec.namespace

    def __init__(
        self,
        words: Optional[Dict[str, Any]] = None,
        chars: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Dict[str, Any]]
    ):
```

The `__DEFAULT_CONFIG` is the default configuration for the `words` section.

### words

The `words` section is passed on to: (__TODO__: align names of `words` and `_WordsFeaturesSpec`)
```python
from biome.text.featurizer import _WordFeaturesSpecs
class _WordFeaturesSpecs:
    def __init__(
        self,
        embedding_dim: int,
        lowercase_tokens: bool = False,
        trainable: bool = True,
        weights_file: Optional[str] = None,
        **extra_params
    ):
```

The `extra_params` must follow the AllenNLP format, that is:
```python
{"indexer": {...}, "embedder": {...}}
```

### chars

The `chars` section is passed on to: (__TODO__: align names of `chars` and `_CharsFeaturesSpec`)
```python
class _CharacterFeaturesSpec:
    def __init__(
        self,
        embedding_dim: int,
        encoder: Dict[str, Any],
        dropout: int = 0.0,
        **extra_params
    ):

```

The `extra_params` must follow the AllenNLP format, that is:
```python
{"indexer": {...}, "embedder": {...}}
```

### kwargs

These must be provided in the following format:
```yaml
name_of_the_feature:
  indexer:  # this is the value of InputFeaturizer.__INDEXER_KEYNAME
    ...
  embedder:  # this is the value of InputFeaturizer.__EMBEDDER_KEYNAME
    ...
another_feature:
  indexer:
    ...
  embedder:
    ...
```

where the `indexer` and `embedder` options (both names are defined in the attributes
`InputFeaturizer.__INDEXER_KEYNAME` and `InputFeaturizer.__EMBEDDER_KEYNAME`)
can be looked up in the
`allennlp.data.token_indexers` and the
`allennlp.modules.token_embedders.TokenEmbedder` classes, respectively
(depending on the specified `type`).

## head

Via the `biome.text.modules.heads.defs.TaskHeadSpec` this section is first passed on to the class:
```python
from biome.text.modules.specs import ComponentConfiguration
class ComponentSpec(Generic[T], FromParams):
    def __init__(self, **config):
        self._config = config or {}
        self._layer_class = self.__resolve_layer_class(self._config.get("type"))

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        return cls(**params.as_dict())

```
Here it uses the `type` key of the section to figure out, which of the Child or Grandchild classes of
`biome.text.modules.heads.defs.TaskHead` to pass on the rest of the section.
The `type` value has to be the name of the corresponding Child/Grandchild class of `TaskHead`
and the class has to be _registered_.
The registering is done in the `biome.text.modules.heads.__init__.py`.

Side note: The Child/Grandchild class has a `model` argument in its `__init__` that you do not have to specify
in the `head` section of the your yaml config.

The arguments for the TaskHead Child/Grandchild class are usually basic python types or AllenNLP classes.
The instantiation of these classes follow _almost_ the standard AllenNLP way: the `type` option specifies the Child class
and the rest of the options are passed on to its init signature, except the "input dimension".
In `biome.text.modules.specs.defs._find_input_attribute` we try to figure out the argument name for
the input dimension and add it on the fly.

## tokenizer

This section is used to init a
```python
from biome.text.configuration import TokenizerConfiguration
class TokenizerConfiguration(FromParams):
    def __init__(
        self,
        lang: str = "en",
        skip_empty_tokens: bool = False,
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
        text_cleaning: Optional[Dict[str, Any]] = None,
        segment_sentences: Union[bool, Dict[str, Any]] = False,
```

The arguments are transformed to
When calling `TokenizerConfiguration.compile` the arguments of the init are embedded in an `allennlp.common.Params`
and passed on to the `biome.text.tokenizer.Tokenizer`.

The `text_cleaning` section is passed on to the
`biome.text.text_cleaning.DefaultTextCleaning` class.
Its `rules` argument can be a list containing the names of the `TextCleaningRule`s in
`biome.text.text_cleaning.py`.

## encoder

This is passed on to a `biome.text.modules.encoders.__init__.Encoder`, which is a
`biome.text.modules.specs.allennlp_specs.Seq2SeqEncoderSpec`.

The options for this section follow _almost_ the standard AllenNLP way for a
`allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder`: the `type` option specifies the Child class
and the rest of the options are passed on to its init signature, except the "input dimension".
In `biome.text.modules.specs.defs._find_input_attribute` we try to figure out the argument name for
the input dimension and add it on the fly.


## Example

This would be an example of a `pipeline.yml`
```yaml
tokenizer:
    text_cleaning:
        rules:
            - strip_spaces

features:
    words:
        embedding_dim: 100
        lowercase_tokens: true
    chars:
        embedding_dim: 8
        encoder:
            type: cnn
            num_filters: 50
            ngram_filter_sizes: [ 4 ]
        dropout: 0.2

encoder:
    hidden_size: 10
    num_layers: 2
    dropout: 0.5
    type: rnn

head:
    type: TextClassification
    labels:
        - duplicate
        - not_duplicate
    pooler:
        type: boe
```

# Datasources

TODO
```yaml
path: ../data/business.cat.10K.csv
format: csv
```

# Training

TODO
```yaml

```
