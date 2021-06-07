(window.webpackJsonp=window.webpackJsonp||[]).push([[61],{468:function(e,t,o){"use strict";o.r(t);var a=o(26),i=Object(a.a)({},(function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[o("h1",{attrs:{id:"nlp-tasks"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#nlp-tasks"}},[e._v("#")]),e._v(" NLP Tasks")]),e._v(" "),o("p",[e._v("In "),o("em",[e._v("biome.text")]),e._v(" NLP tasks are defined via "),o("code",[e._v("TaskHead")]),e._v(" classes.")]),e._v(" "),o("p",[e._v("This section gives a summary of the library's main heads and tasks.")]),e._v(" "),o("h2",{attrs:{id:"textclassification"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#textclassification"}},[e._v("#")]),e._v(" TextClassification")]),e._v(" "),o("p",[o("strong",[e._v("Tutorials")]),e._v(": "),o("RouterLink",{attrs:{to:"/documentation/tutorials/Training_a_text_classifier.html"}},[e._v("Training a short text classifier of German business names")])],1),e._v(" "),o("p",[o("strong",[e._v("NLP tasks")]),e._v(": text classification, sentiment analysis, entity typing, relation classification.")]),e._v(" "),o("p",[o("strong",[e._v("Input")]),e._v(": "),o("code",[e._v("text")]),e._v(": a single field or a concatenation of input fields.")]),e._v(" "),o("p",[o("strong",[e._v("Output")]),e._v(": "),o("code",[e._v("label")]),e._v(" by default, a probability distribution over labels except if "),o("code",[e._v("multilabel")]),e._v(" is enabled for multi-label classification problems.")]),e._v(" "),o("p",[o("strong",[e._v("Main parameters")]),e._v(":")]),e._v(" "),o("p",[o("code",[e._v("pooler")]),e._v(": a "),o("code",[e._v("Seq2VecEncoderConfiguration")]),e._v(" to pool a sequence of encoded word/char vectors into a single vector representing the input text")]),e._v(" "),o("p",[e._v("See "),o("RouterLink",{attrs:{to:"/api/biome/text/modules/heads/classification/text_classification.html#textclassification"}},[e._v("TextClassification API")]),e._v(" for more details.")],1),e._v(" "),o("h2",{attrs:{id:"recordclassification"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#recordclassification"}},[e._v("#")]),e._v(" RecordClassification")]),e._v(" "),o("p",[o("strong",[e._v("NLP tasks")]),e._v(": text classification, sentiment analysis, entity typing, relation classification and semi-structured data classification problems with product, customer data, etc.")]),e._v(" "),o("p",[o("strong",[e._v("Input")]),e._v(": "),o("code",[e._v("document")]),e._v(": a list of fields.")]),e._v(" "),o("p",[o("strong",[e._v("Output")]),e._v(": "),o("code",[e._v("labels")]),e._v(" by default, a probability distribution over labels except if "),o("code",[e._v("multilabel")]),e._v(" is enabled for multi-label classification problems.")]),e._v(" "),o("p",[o("strong",[e._v("Main parameters")]),e._v(":")]),e._v(" "),o("p",[o("code",[e._v("record_keys")]),e._v(": field keys to be used as input features to the model, e.g., name, first_name, body, subject, etc.")]),e._v(" "),o("p",[o("code",[e._v("tokens_pooler")]),e._v(": a "),o("code",[e._v("Seq2VecEncoderConfiguration")]),e._v(" to pool a sequence of encoded word/char vectors "),o("strong",[e._v("for each field")]),e._v(" into a single vector representing the field.")]),e._v(" "),o("p",[o("code",[e._v("fields_encoder")]),e._v(": a "),o("code",[e._v("Seq2SeqEncoderConfiguration")]),e._v(" to encode a sequence of field vectors.")]),e._v(" "),o("p",[o("code",[e._v("fields_pooler")]),e._v(": a "),o("code",[e._v("Seq2VecEncoderConfiguration")]),e._v(" to pool a sequence of encoded field vectors into a single vector representing the whole document/record.")]),e._v(" "),o("p",[e._v("See "),o("RouterLink",{attrs:{to:"/api/biome/text/modules/heads/classification/record_classification.html#recordclassification"}},[e._v("RecordClassification API")]),e._v(" for more details.")],1),e._v(" "),o("h2",{attrs:{id:"recordpairclassification"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#recordpairclassification"}},[e._v("#")]),e._v(" RecordPairClassification")]),e._v(" "),o("p",[o("strong",[e._v("NLP tasks")]),e._v(": Classify the relation between a pair of structured data. For example, do two sets of customer data belong to the same customer or not.")]),e._v(" "),o("p",[o("strong",[e._v("Input")]),e._v(": "),o("code",[e._v("record1")]),e._v(", "),o("code",[e._v("record2")]),e._v(". Two dictionaries that should share the same keys, preferably in the same order.")]),e._v(" "),o("p",[o("strong",[e._v("Output")]),e._v(": "),o("code",[e._v("labels")]),e._v(". By default, a probability distribution over labels except if "),o("code",[e._v("multilabel")]),e._v(" is enabled for multi-label classification problems.")]),e._v(" "),o("p",[o("strong",[e._v("Main parameters")]),e._v(":")]),e._v(" "),o("p",[o("code",[e._v("field_encoder")]),e._v(": A "),o("code",[e._v("Seq2VecEncoder")]),e._v(" to encode and pool the single dictionary items of both inputs. It takes both, the key and the value, into account.")]),e._v(" "),o("p",[o("code",[e._v("record_encoder")]),e._v(": A "),o("code",[e._v("Seq2SeqEncoder")]),e._v(" to contextualize the encoded dictionary items within its record.")]),e._v(" "),o("p",[o("code",[e._v("matcher_forward")]),e._v(": A "),o("code",[e._v("BiMPMMatching")]),e._v(" layer for the (optionally only forward) record encoder layer.")]),e._v(" "),o("p",[o("code",[e._v("aggregator")]),e._v(": A "),o("code",[e._v("Seq2VecEncoder")]),e._v(" to pool the output of the matching layers.")]),e._v(" "),o("p",[e._v("See the "),o("RouterLink",{attrs:{to:"/api/biome/text/modules/heads/classification/record_pair_classification.html"}},[e._v("RecordPairClassification API")]),e._v(" for more details.")],1),e._v(" "),o("h2",{attrs:{id:"tokenclassification"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#tokenclassification"}},[e._v("#")]),e._v(" TokenClassification")]),e._v(" "),o("p",[o("strong",[e._v("Tutorials")]),e._v(": "),o("RouterLink",{attrs:{to:"/documentation/tutorials/Training_a_sequence_tagger_for_Slot_Filling.html"}},[e._v("Training a sequence tagger for Slot Filling")])],1),e._v(" "),o("p",[o("strong",[e._v("NLP tasks")]),e._v(": NER, Slot filling, Part of speech tagging.")]),e._v(" "),o("p",[o("strong",[e._v("Input")]),e._v(": "),o("code",[e._v("text")]),e._v(": "),o("strong",[e._v("pretokenized text")]),e._v(" as a list of tokens.")]),e._v(" "),o("p",[o("strong",[e._v("Output")]),e._v(": "),o("code",[e._v("labels")]),e._v(": one label for each token according to the "),o("code",[e._v("label_encoding")]),e._v(" scheme defined in the head (e.g., BIO).")]),e._v(" "),o("p",[o("strong",[e._v("Main parameters")]),e._v(":")]),e._v(" "),o("p",[o("code",[e._v("feedforward")]),e._v(": feed-forward layer to be applied after token encoding.")]),e._v(" "),o("p",[e._v("See "),o("RouterLink",{attrs:{to:"/api/biome/text/modules/heads/token_classification.html#tokenclassification"}},[e._v("TokenClassification API")]),e._v(" for more details.")],1),e._v(" "),o("h2",{attrs:{id:"languagemodelling"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#languagemodelling"}},[e._v("#")]),e._v(" LanguageModelling")]),e._v(" "),o("p",[o("strong",[e._v("NLP tasks")]),e._v(": Pre-training, word-level next token language model.")]),e._v(" "),o("p",[o("strong",[e._v("Input")]),e._v(": "),o("code",[e._v("text")]),e._v(": a single field or a concatenation of input fields.")]),e._v(" "),o("p",[o("strong",[e._v("Output")]),e._v(": contextualized word vectors.")]),e._v(" "),o("p",[o("strong",[e._v("Main parameters")]),e._v(":")]),e._v(" "),o("p",[o("code",[e._v("dropout")]),e._v(" to be applied after token encoding.")]),e._v(" "),o("p",[e._v("See "),o("RouterLink",{attrs:{to:"/api/biome/text/modules/heads/language_modelling.html#languagemodelling"}},[e._v("LanguageModelling API")]),e._v(" for more details.")],1)])}),[],!1,null,null,null);t.default=i.exports}}]);