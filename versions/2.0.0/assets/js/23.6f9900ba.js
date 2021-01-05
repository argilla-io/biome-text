(window.webpackJsonp=window.webpackJsonp||[]).push([[23],{397:function(e,t,a){"use strict";a.r(t);var n=a(26),s=Object(n.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-configuration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-configuration"}},[e._v("#")]),e._v(" biome.text.configuration "),a("Badge",{attrs:{text:"Module"}})],1),e._v(" "),a("div"),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"featuresconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#featuresconfiguration"}},[e._v("#")]),e._v(" FeaturesConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("FeaturesConfiguration")]),e._v(" ("),e._v("\n    "),a("span",[e._v("word: Union["),a("a",{attrs:{title:"biome.text.features.WordFeatures",href:"features.html#biome.text.features.WordFeatures"}},[e._v("WordFeatures")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("char: Union["),a("a",{attrs:{title:"biome.text.features.CharFeatures",href:"features.html#biome.text.features.CharFeatures"}},[e._v("CharFeatures")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("transformers: Union["),a("a",{attrs:{title:"biome.text.features.TransformersFeatures",href:"features.html#biome.text.features.TransformersFeatures"}},[e._v("TransformersFeatures")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Configures the input features of the "),a("code",[e._v("Pipeline")])]),e._v(" "),a("p",[e._v("Use this for defining the features to be used by the model, namely word and character embeddings.")]),e._v(" "),a("p",[e._v(":::tip\nIf you do not pass in either of the parameters ("),a("code",[e._v("word")]),e._v(" or "),a("code",[e._v("char")]),e._v("),\nyour pipeline will be setup with a default word feature (embedding_dim=50).\n:::")]),e._v(" "),a("p",[e._v("Example:")]),e._v(" "),a("pre",[a("code",{staticClass:"language-python"},[e._v("word = WordFeatures(embedding_dim=100)\nchar = CharFeatures(embedding_dim=16, encoder={'type': 'gru'})\nconfig = FeaturesConfiguration(word, char)\n")])]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("word")])])]),e._v(" "),a("dd",[e._v("The word feature configurations, see "),a("code",[a("a",{attrs:{title:"biome.text.features.WordFeatures",href:"features.html#biome.text.features.WordFeatures"}},[e._v("WordFeatures")])])]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("char")])])]),e._v(" "),a("dd",[e._v("The character feature configurations, see "),a("code",[a("a",{attrs:{title:"biome.text.features.CharFeatures",href:"features.html#biome.text.features.CharFeatures"}},[e._v("CharFeatures")])])]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("transformers")])])]),e._v(" "),a("dd",[e._v("The transformers feature configuration, see "),a("code",[a("a",{attrs:{title:"biome.text.features.TransformersFeatures",href:"features.html#biome.text.features.TransformersFeatures"}},[e._v("TransformersFeatures")])]),e._v("\nA word-level representation of the "),a("a",{attrs:{href:"https://huggingface.co/models"}},[e._v("transformer")]),e._v(" models using AllenNLP's")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-params"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-params"}},[e._v("#")]),e._v(" from_params "),a("Badge",{attrs:{text:"Static method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("from_params")]),e._v(" ("),e._v("\n  params: allennlp.common.params.Params,\n  **extras,\n)  -> "),a("a",{attrs:{title:"biome.text.configuration.FeaturesConfiguration",href:"#biome.text.configuration.FeaturesConfiguration"}},[e._v("FeaturesConfiguration")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("This is the automatic implementation of "),a("code",[e._v("from_params")]),e._v(". Any class that subclasses\n"),a("code",[e._v("FromParams")]),e._v(" (or "),a("code",[e._v("Registrable")]),e._v(", which itself subclasses "),a("code",[e._v("FromParams")]),e._v(') gets this\nimplementation for free.\nIf you want your class to be instantiated from params in the\n"obvious" way – pop off parameters and hand them to your constructor with the same names –\nthis provides that functionality.')]),e._v(" "),a("p",[e._v("If you need more complex logic in your from "),a("code",[e._v("from_params")]),e._v(" method, you'll have to implement\nyour own method that overrides this one.")]),e._v(" "),a("p",[e._v("The "),a("code",[e._v("constructor_to_call")]),e._v(" and "),a("code",[e._v("constructor_to_inspect")]),e._v(" arguments deal with a bit of\nredirection that we do.\nWe allow you to register particular "),a("code",[e._v("@classmethods")]),e._v(" on a class as\nthe constructor to use for a registered name.\nThis lets you, e.g., have a single\n"),a("code",[e._v("Vocabulary")]),e._v(" class that can be constructed in two different ways, with different names\nregistered to each constructor.\nIn order to handle this, we need to know not just the class\nwe're trying to construct ("),a("code",[e._v("cls")]),e._v("), but also what method we should inspect to find its\narguments ("),a("code",[e._v("constructor_to_inspect")]),e._v("), and what method to call when we're done constructing\narguments ("),a("code",[e._v("constructor_to_call")]),e._v(").\nThese two methods are the same when you've used a\n"),a("code",[e._v("@classmethod")]),e._v(" as your constructor, but they are "),a("code",[e._v("different")]),e._v(" when you use the default\nconstructor (because you inspect "),a("code",[e._v("__init__")]),e._v(", but call "),a("code",[e._v("cls()")]),e._v(").")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"instance-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[e._v("#")]),e._v(" Instance variables")]),e._v("\n")]),e._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.configuration.FeaturesConfiguration.configured_namespaces"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("configured_namespaces")]),e._v(" : List[str]")])]),e._v(" "),a("dd",[a("p",[e._v("Return the namespaces of the features that are configured")])])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"compile-embedder"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#compile-embedder"}},[e._v("#")]),e._v(" compile_embedder "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("compile_embedder")]),e._v(" ("),e._v("\n  self,\n  vocab: allennlp.data.vocabulary.Vocabulary,\n)  -> allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates the embedder based on the configured input features")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("vocab")])])]),e._v(" "),a("dd",[e._v("The vocabulary for which to create the embedder")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("embedder")])]),e._v(" "),a("dd",[e._v(" ")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"compile-featurizer"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#compile-featurizer"}},[e._v("#")]),e._v(" compile_featurizer "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("compile_featurizer")]),e._v(" ("),e._v("\n  self,\n  tokenizer: "),a("a",{attrs:{title:"biome.text.tokenizer.Tokenizer",href:"tokenizer.html#biome.text.tokenizer.Tokenizer"}},[e._v("Tokenizer")]),e._v(",\n)  -> "),a("a",{attrs:{title:"biome.text.featurizer.InputFeaturizer",href:"featurizer.html#biome.text.featurizer.InputFeaturizer"}},[e._v("InputFeaturizer")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates the featurizer based on the configured input features")]),e._v(" "),a("p",[e._v(":::tip\nIf you are creating configurations programmatically\nuse this method to check that you provided a valid configuration.\n:::")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("tokenizer")])])]),e._v(" "),a("dd",[e._v("Tokenizer used for this featurizer")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("featurizer")])]),e._v(" "),a("dd",[e._v("The configured "),a("code",[e._v("InputFeaturizer")])])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"tokenizerconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tokenizerconfiguration"}},[e._v("#")]),e._v(" TokenizerConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("TokenizerConfiguration")]),e._v(" ("),e._v("\n    "),a("span",[e._v("lang: str = 'en'")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("max_sequence_length: int = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("max_nr_of_sentences: int = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("text_cleaning: Union[Dict[str, Any], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("segment_sentences: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("use_spacy_tokens: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("remove_space_tokens: bool = True")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("start_tokens: Union[List[str], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("end_tokens: Union[List[str], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("use_transformers: Union[bool, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("transformers_kwargs: Union[Dict, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Configures the "),a("code",[e._v("Tokenizer")])]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("lang")])])]),e._v(" "),a("dd",[e._v("The "),a("a",{attrs:{href:"https://spacy.io/api/tokenizer"}},[e._v("spaCy model used")]),e._v(' for tokenization is language dependent.\nFor optimal performance, specify the language of your input data (default: "en").')]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("max_sequence_length")])])]),e._v(" "),a("dd",[e._v("Maximum length in characters for input texts truncated with "),a("code",[e._v("[:max_sequence_length]")]),e._v(" after "),a("code",[e._v("TextCleaning")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("max_nr_of_sentences")])])]),e._v(" "),a("dd",[e._v("Maximum number of sentences to keep when using "),a("code",[e._v("segment_sentences")]),e._v(" truncated with "),a("code",[e._v("[:max_sequence_length]")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("text_cleaning")])])]),e._v(" "),a("dd",[e._v("A "),a("code",[e._v("TextCleaning")]),e._v(" configuration with pre-processing rules for cleaning up and transforming raw input text.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("segment_sentences")])])]),e._v(" "),a("dd",[e._v("Whether to segment input texts into sentences.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("use_spacy_tokens")])])]),e._v(" "),a("dd",[e._v("If True, the tokenized token list contains spacy tokens instead of allennlp tokens")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("remove_space_tokens")])])]),e._v(" "),a("dd",[e._v("If True, all found space tokens will be removed from the final token list.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("start_tokens")])])]),e._v(" "),a("dd",[e._v("A list of token strings to the sequence before tokenized input text.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("end_tokens")])])]),e._v(" "),a("dd",[e._v("A list of token strings to the sequence after tokenized input text.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("use_transformers")])])]),e._v(" "),a("dd",[e._v("If true, we will use a transformers tokenizer from HuggingFace and disregard all other parameters above.\nIf you specify any of the above parameters you want to set this to false.\nIf None, we automatically choose the right value based on your feature and head configuration.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("transformers_kwargs")])])]),e._v(" "),a("dd",[e._v("This dict is passed on to AllenNLP's "),a("code",[e._v("PretrainedTransformerTokenizer")]),e._v(".\nIf no "),a("code",[e._v("model_name")]),e._v(" key is provided, we will infer one from the features configuration.")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"pipelineconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pipelineconfiguration"}},[e._v("#")]),e._v(" PipelineConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("PipelineConfiguration")]),e._v(" ("),e._v("\n    "),a("span",[e._v("name: str")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("head: "),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHeadConfiguration",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHeadConfiguration"}},[e._v("TaskHeadConfiguration")])]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("features: Union["),a("a",{attrs:{title:"biome.text.configuration.FeaturesConfiguration",href:"#biome.text.configuration.FeaturesConfiguration"}},[e._v("FeaturesConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("tokenizer: Union["),a("a",{attrs:{title:"biome.text.configuration.TokenizerConfiguration",href:"#biome.text.configuration.TokenizerConfiguration"}},[e._v("TokenizerConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("encoder: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration",href:"modules/configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration"}},[e._v("Seq2SeqEncoderConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Creates a "),a("code",[e._v("Pipeline")]),e._v(" configuration")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("name")])])]),e._v(" "),a("dd",[e._v("The "),a("code",[e._v("name")]),e._v(" for our pipeline")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("features")])])]),e._v(" "),a("dd",[e._v("The input "),a("code",[e._v("features")]),e._v(" to be used by the model pipeline. We define this using a "),a("code",[a("a",{attrs:{title:"biome.text.configuration.FeaturesConfiguration",href:"#biome.text.configuration.FeaturesConfiguration"}},[e._v("FeaturesConfiguration")])]),e._v(" object.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("head")])])]),e._v(" "),a("dd",[e._v("The "),a("code",[e._v("head")]),e._v(" for the task, e.g., a LanguageModelling task, using a "),a("code",[e._v("TaskHeadConfiguration")]),e._v(" object.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("tokenizer")])])]),e._v(" "),a("dd",[e._v("The "),a("code",[e._v("tokenizer")]),e._v(" defined with a "),a("code",[a("a",{attrs:{title:"biome.text.configuration.TokenizerConfiguration",href:"#biome.text.configuration.TokenizerConfiguration"}},[e._v("TokenizerConfiguration")])]),e._v(" object.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("encoder")])])]),e._v(" "),a("dd",[e._v("The core text seq2seq "),a("code",[e._v("encoder")]),e._v(" of our model using a "),a("code",[e._v("Seq2SeqEncoderConfiguration")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-3"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-yaml"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-yaml"}},[e._v("#")]),e._v(" from_yaml "),a("Badge",{attrs:{text:"Static method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("from_yaml")]),e._v("("),a("span",[e._v("path: str) -> "),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"#biome.text.configuration.PipelineConfiguration"}},[e._v("PipelineConfiguration")])]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates a pipeline configuration from a config yaml file")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("path")])])]),e._v(" "),a("dd",[e._v("The path to a YAML configuration file")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("pipeline_configuration")])]),e._v(" "),a("dd",[e._v(" ")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-dict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-dict"}},[e._v("#")]),e._v(" from_dict "),a("Badge",{attrs:{text:"Static method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("from_dict")]),e._v("("),a("span",[e._v("config_dict: dict) -> "),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"#biome.text.configuration.PipelineConfiguration"}},[e._v("PipelineConfiguration")])]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates a pipeline configuration from a config dictionary")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("config_dict")])])]),e._v(" "),a("dd",[e._v("A configuration dictionary")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("pipeline_configuration")])]),e._v(" "),a("dd",[e._v(" ")])])])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"as-dict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#as-dict"}},[e._v("#")]),e._v(" as_dict "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("as_dict")]),e._v("("),a("span",[e._v("self) -> Dict[str, Any]")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Returns the configuration as dictionary")]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("config")])]),e._v(" "),a("dd",[e._v(" ")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"to-yaml"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#to-yaml"}},[e._v("#")]),e._v(" to_yaml "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("to_yaml")]),e._v(" ("),e._v("\n  self,\n  path: str,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Saves the pipeline configuration to a yaml formatted file")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("path")])])]),e._v(" "),a("dd",[e._v("Path to the output file")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"build-tokenizer"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#build-tokenizer"}},[e._v("#")]),e._v(" build_tokenizer "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("build_tokenizer")]),e._v("("),a("span",[e._v("self) -> "),a("a",{attrs:{title:"biome.text.tokenizer.Tokenizer",href:"tokenizer.html#biome.text.tokenizer.Tokenizer"}},[e._v("Tokenizer")])]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Build the pipeline tokenizer")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"build-featurizer"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#build-featurizer"}},[e._v("#")]),e._v(" build_featurizer "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("build_featurizer")]),e._v("("),a("span",[e._v("self) -> "),a("a",{attrs:{title:"biome.text.featurizer.InputFeaturizer",href:"featurizer.html#biome.text.featurizer.InputFeaturizer"}},[e._v("InputFeaturizer")])]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates the pipeline featurizer")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"build-embedder"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#build-embedder"}},[e._v("#")]),e._v(" build_embedder "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("build_embedder")]),e._v(" ("),e._v("\n  self,\n  vocab: allennlp.data.vocabulary.Vocabulary,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Build the pipeline embedder for aiven dictionary")])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"trainerconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#trainerconfiguration"}},[e._v("#")]),e._v(" TrainerConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("TrainerConfiguration")]),e._v(" ("),e._v("\n    "),a("span",[e._v("optimizer: Dict[str, Any] = <factory>")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("validation_metric: str = '-loss'")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("patience: Union[int, NoneType] = 2")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("num_epochs: int = 20")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("cuda_device: Union[int, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("grad_norm: Union[float, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("grad_clipping: Union[float, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("learning_rate_scheduler: Union[Dict[str, Any], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("momentum_scheduler: Union[Dict[str, Any], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("moving_average: Union[Dict[str, Any], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("use_amp: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("num_serialized_models_to_keep: int = 1")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("batch_size: Union[int, NoneType] = 16")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("data_bucketing: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("batches_per_epoch: Union[int, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("random_seed: Union[int, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Configures the training of a pipeline")]),e._v(" "),a("p",[e._v("It is passed on to the "),a("code",[e._v("Pipeline.train")]),e._v(" method. Doc strings mainly provided by\n"),a("a",{attrs:{href:"https://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects"}},[e._v("AllenNLP")])]),e._v(" "),a("h2",{attrs:{id:"attributes"}},[e._v("Attributes")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("optimizer")])])]),e._v(" "),a("dd",[a("a",{attrs:{href:"https://pytorch.org/docs/stable/optim.html"}},[e._v("Pytorch optimizers")]),e._v("\nthat can be constructed via the AllenNLP configuration framework")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("validation_metric")])])]),e._v(" "),a("dd",[e._v('Validation metric to measure for whether to stop training using patience\nand whether to serialize an is_best model each epoch.\nThe metric name must be prepended with either "+" or "-",\nwhich specifies whether the metric is an increasing or decreasing function.')]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("patience")])])]),e._v(" "),a("dd",[e._v("Number of epochs to be patient before early stopping:\nthe training is stopped after "),a("code",[e._v("patience")]),e._v(" epochs with no improvement.\nIf given, it must be > 0. If "),a("code",[e._v("None")]),e._v(", early stopping is disabled.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("num_epochs")])])]),e._v(" "),a("dd",[e._v("Number of training epochs")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("cuda_device")])])]),e._v(" "),a("dd",[e._v("An integer specifying the CUDA device to use for this process. If -1, the CPU is used.\nBy default (None) we will automatically use a CUDA device if one is available.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("grad_norm")])])]),e._v(" "),a("dd",[e._v("If provided, gradient norms will be rescaled to have a maximum of this value.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("grad_clipping")])])]),e._v(" "),a("dd",[e._v("If provided, gradients will be clipped during the backward pass to have an (absolute) maximum of this value.\nIf you are getting "),a("code",[e._v("NaN")]),e._v("s in your gradients during training that are not solved by using grad_norm,\nyou may need this.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("learning_rate_scheduler")])])]),e._v(" "),a("dd",[e._v("If specified, the learning rate will be decayed with respect to this schedule at the end of each epoch\n(or batch, if the scheduler implements the step_batch method).\nIf you use "),a("code",[e._v("torch.optim.lr_scheduler.ReduceLROnPlateau")]),e._v(", this will use the "),a("code",[e._v("validation_metric")]),e._v(" provided\nto determine if learning has plateaued.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("momentum_scheduler")])])]),e._v(" "),a("dd",[e._v("If specified, the momentum will be updated at the end of each batch or epoch according to the schedule.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("moving_average")])])]),e._v(" "),a("dd",[e._v("If provided, we will maintain moving averages for all parameters.\nDuring training, we employ a shadow variable for each parameter, which maintains the moving average.\nDuring evaluation, we backup the original parameters and assign the moving averages to corresponding parameters.\nBe careful that when saving the checkpoint, we will save the moving averages of parameters.\nThis is necessary because we want the saved model to perform as well as the validated model if we load it later.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("batch_size")])])]),e._v(" "),a("dd",[e._v("Size of the batch.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("data_bucketing")])])]),e._v(" "),a("dd",[e._v("If enabled, try to apply data bucketing over training batches.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("batches_per_epoch")])])]),e._v(" "),a("dd",[e._v('Determines the number of batches after which a training epoch ends.\nIf the number is smaller than the total amount of batches in your training data,\nthe second "epoch" will take off where the first "epoch" ended.\nIf this is '),a("code",[e._v("None")]),e._v(", then an epoch is set to be one full pass through your training data.\nThis is useful if you want to evaluate your data more frequently on your validation data set during training.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("random_seed")])])]),e._v(" "),a("dd",[e._v("Seed for the underlying random number generators.\nIf None, we take the random seeds provided by AllenNLP's "),a("code",[e._v("prepare_environment")]),e._v(" method.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("use_amp")])])]),e._v(" "),a("dd",[e._v("If "),a("code",[e._v("True")]),e._v(", we'll train using "),a("a",{attrs:{href:"https://pytorch.org/docs/stable/amp.html"}},[e._v("Automatic Mixed Precision")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("num_serialized_models_to_keep")])])]),e._v(" "),a("dd",[e._v("Number of previous model checkpoints to retain.\nDefault is to keep 1 checkpoint.\nA value of None or -1 means all checkpoints will be kept.")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"to-allennlp-trainer"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#to-allennlp-trainer"}},[e._v("#")]),e._v(" to_allennlp_trainer "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("to_allennlp_trainer")]),e._v("("),a("span",[e._v("self) -> Dict[str, Any]")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Returns a configuration dict formatted for AllenNLP's trainer")]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("allennlp_trainer_config")])]),e._v(" "),a("dd",[e._v(" ")])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"vocabularyconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#vocabularyconfiguration"}},[e._v("#")]),e._v(" VocabularyConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("VocabularyConfiguration")]),e._v(" ("),e._v("\n    "),a("span",[e._v("datasets: List["),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[e._v("Dataset")]),e._v("]")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("min_count: Dict[str, int] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("max_vocab_size: Union[int, Dict[str, int]] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("pretrained_files: Union[Dict[str, str], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("only_include_pretrained_words: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("tokens_to_add: Dict[str, List[str]] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("min_pretrained_embeddings: Dict[str, int] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Configures a "),a("code",[e._v("Vocabulary")]),e._v(" before it gets created from the data")]),e._v(" "),a("p",[e._v("Use this to configure a Vocabulary using specific arguments from "),a("code",[e._v("allennlp.data.Vocabulary")])]),e._v(" "),a("p",[e._v("See "),a("a",{attrs:{href:"https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary]"}},[e._v("AllenNLP Vocabulary docs")])]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("datasets")])])]),e._v(" "),a("dd",[e._v("List of datasets from which to create the vocabulary")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("min_count")])])]),e._v(" "),a("dd",[e._v("Minimum number of appearances of a token to be included in the vocabulary.\nThe key in the dictionary refers to the namespace of the input feature")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("max_vocab_size")])])]),e._v(" "),a("dd",[e._v("If you want to cap the number of tokens in your vocabulary, you can do so with this\nparameter.\nIf you specify a single integer, every namespace will have its vocabulary fixed\nto be no larger than this.\nIf you specify a dictionary, then each namespace in the\n"),a("code",[e._v("counter")]),e._v(" can have a separate maximum vocabulary size. Any missing key will have a value\nof "),a("code",[e._v("None")]),e._v(", which means no cap on the vocabulary size.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("pretrained_files")])])]),e._v(" "),a("dd",[e._v("If provided, this map specifies the path to optional pretrained embedding files for each\nnamespace. This can be used to either restrict the vocabulary to only words which appear\nin this file, or to ensure that any words in this file are included in the vocabulary\nregardless of their count, depending on the value of "),a("code",[e._v("only_include_pretrained_words")]),e._v(".\nWords which appear in the pretrained embedding file but not in the data are NOT included\nin the Vocabulary.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("only_include_pretrained_words")])])]),e._v(" "),a("dd",[e._v("Only include tokens present in pretrained_files")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("tokens_to_add")])])]),e._v(" "),a("dd",[e._v("A list of tokens to add to the corresponding namespace of the vocabulary,\neven if they are not present in the "),a("code",[e._v("datasets")])]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("min_pretrained_embeddings")])])]),e._v(" "),a("dd",[e._v("Minimum number of lines to keep from pretrained_files, even for tokens not appearing in the sources.")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"build-vocab"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#build-vocab"}},[e._v("#")]),e._v(" build_vocab "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("build_vocab")]),e._v(" ("),e._v("\n  self,\n  pipeline: Pipeline,\n  lazy: bool = False,\n)  -> allennlp.data.vocabulary.Vocabulary\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Build the configured vocabulary")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("pipeline")])])]),e._v(" "),a("dd",[e._v("The pipeline used to create the instances from which the vocabulary is built.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("lazy")])])]),e._v(" "),a("dd",[e._v("If true, instances are lazily loaded from disk, otherwise they are loaded into memory.")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("vocab")])]),e._v(" "),a("dd",[e._v(" ")])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"findlrconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#findlrconfiguration"}},[e._v("#")]),e._v(" FindLRConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("FindLRConfiguration")]),e._v(" ("),e._v("\n    "),a("span",[e._v("start_lr: float = 1e-05")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("end_lr: float = 10")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("num_batches: int = 100")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("linear_steps: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("stopping_factor: Union[float, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("A configuration for finding the learning rate via "),a("code",[e._v("Pipeline.find_lr()")]),e._v(".")]),e._v(" "),a("p",[e._v("The "),a("code",[e._v("Pipeline.find_lr()")]),e._v(" method increases the learning rate from "),a("code",[e._v("start_lr")]),e._v(" to "),a("code",[e._v("end_lr")]),e._v(" recording the losses.")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("start_lr")])])]),e._v(" "),a("dd",[e._v("The learning rate to start the search.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("end_lr")])])]),e._v(" "),a("dd",[e._v("The learning rate upto which search is done.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("num_batches")])])]),e._v(" "),a("dd",[e._v("Number of batches to run the learning rate finder.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("linear_steps")])])]),e._v(" "),a("dd",[e._v("Increase learning rate linearly if False exponentially.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("stopping_factor")])])]),e._v(" "),a("dd",[e._v("Stop the search when the current loss exceeds the best loss recorded by\nmultiple of stopping factor. If "),a("code",[e._v("None")]),e._v(" search proceeds till the "),a("code",[e._v("end_lr")])])])])}),[],!1,null,null,null);t.default=s.exports}}]);