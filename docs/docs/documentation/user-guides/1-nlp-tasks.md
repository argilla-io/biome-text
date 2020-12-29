# NLP Tasks

In *biome.text* NLP tasks are defined via ``TaskHead`` classes.

This section gives a summary of the library's main heads and tasks.

## TextClassification

**Tutorials**: [Training a short text classifier of German business names](../tutorials/Training_a_text_classifier.md)

**NLP tasks**: text classification, sentiment analysis, entity typing, relation classification.

**Input**: `text`: a single field or a concatenation of input fields.

**Output**: `label` by default, a probability distribution over labels except if `multilabel` is enabled for multi-label classification problems.

**Main parameters**:

`pooler`: a `Seq2VecEncoderConfiguration` to pool a sequence of encoded word/char vectors into a single vector representing the input text


See [TextClassification API](../../api/biome/text/modules/heads/classification/text_classification.md#textclassification) for more details.

## RecordClassification

**NLP tasks**: text classification, sentiment analysis, entity typing, relation classification and semi-structured data classification problems with product, customer data, etc.

**Input**: `document`: a list of fields.

**Output**: `labels` by default, a probability distribution over labels except if `multilabel` is enabled for multi-label classification problems.

**Main parameters**:

`record_keys`: field keys to be used as input features to the model, e.g., name, first_name, body, subject, etc.

`tokens_pooler`: a `Seq2VecEncoderConfiguration` to pool a sequence of encoded word/char vectors **for each field** into a single vector representing the field.

`fields_encoder`: a `Seq2SeqEncoderConfiguration` to encode a sequence of field vectors.

`fields_pooler`: a `Seq2VecEncoderConfiguration` to pool a sequence of encoded field vectors into a single vector representing the whole document/record.

See [RecordClassification API](../../api/biome/text/modules/heads/classification/record_classification.md#recordclassification) for more details.

## RecordPairClassification

**NLP tasks**: Classify the relation between a pair of structured data. For example, do two sets of customer data belong to the same customer or not.

**Input**: `record1`, `record2`. Two dictionaries that should share the same keys, preferably in the same order.

**Output**: `labels`. By default, a probability distribution over labels except if `multilabel` is enabled for multi-label classification problems.

**Main parameters**:

`field_encoder`: A `Seq2VecEncoder` to encode and pool the single dictionary items of both inputs. It takes both, the key and the value, into account.

`record_encoder`: A `Seq2SeqEncoder` to contextualize the encoded dictionary items within its record.

`matcher_forward`: A `BiMPMMatching` layer for the (optionally only forward) record encoder layer.

`aggregator`: A `Seq2VecEncoder` to pool the output of the matching layers.

See the [RecordPairClassification API](../../api/biome/text/modules/heads/classification/record_pair_classification.md) for more details.

##  TokenClassification

**Tutorials**: [Training a sequence tagger for Slot Filling](../tutorials/Training_a_sequence_tagger_for_Slot_Filling.md)

**NLP tasks**: NER, Slot filling, Part of speech tagging.

**Input**: `text`: **pretokenized text** as a list of tokens.

**Output**: `labels`: one label for each token according to the `label_encoding` scheme defined in the head (e.g., BIO).

**Main parameters**:

`feedforward`: feed-forward layer to be applied after token encoding.

See [TokenClassification API](../../api/biome/text/modules/heads/token_classification.md#tokenclassification) for more details.

##  LanguageModelling

**NLP tasks**: Pre-training, word-level next token language model.

**Input**: `text`: a single field or a concatenation of input fields.

**Output**: contextualized word vectors.

**Main parameters**:

`dropout` to be applied after token encoding.


See [LanguageModelling API](../../api/biome/text/modules/heads/language_modelling.md#languagemodelling) for more details.
