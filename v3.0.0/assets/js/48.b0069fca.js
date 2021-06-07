(window.webpackJsonp=window.webpackJsonp||[]).push([[48],{454:function(e,t,a){"use strict";a.r(t);var i=a(26),s=Object(i.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-pipeline"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-pipeline"}},[e._v("#")]),e._v(" biome.text.pipeline "),a("Badge",{attrs:{text:"Module"}})],1),e._v(" "),a("div"),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"pipeline"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pipeline"}},[e._v("#")]),e._v(" Pipeline "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("Pipeline")]),e._v(" (model: "),a("a",{attrs:{title:"biome.text.model.PipelineModel",href:"model.html#biome.text.model.PipelineModel"}},[e._v("PipelineModel")]),e._v(", config: "),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[e._v("PipelineConfiguration")]),e._v(")"),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Manages NLP models configuration and actions.")]),e._v(" "),a("p",[e._v("Use "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[e._v("Pipeline")])]),e._v(" for creating new models from a configuration or loading a pretrained model.")]),e._v(" "),a("p",[e._v("Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.")]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-yaml"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-yaml"}},[e._v("#")]),e._v(" from_yaml "),a("Badge",{attrs:{text:"Static method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("from_yaml")]),e._v(" ("),e._v("\n  path: str,\n  vocab_path: Union[str, NoneType] = None,\n)  -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[e._v("Pipeline")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates a pipeline from a config yaml file")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("path")])])]),e._v(" "),a("dd",[e._v("The path to a YAML configuration file")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("vocab_path")])])]),e._v(" "),a("dd",[e._v("If provided, the pipeline vocab will be loaded from this path")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("pipeline")])]),e._v(" "),a("dd",[e._v("A configured pipeline")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-config"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-config"}},[e._v("#")]),e._v(" from_config "),a("Badge",{attrs:{text:"Static method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("from_config")]),e._v(" ("),e._v("\n  config: Union["),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[e._v("PipelineConfiguration")]),e._v(", dict],\n  vocab_path: Union[str, NoneType] = None,\n)  -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[e._v("Pipeline")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates a pipeline from a "),a("code",[e._v("PipelineConfiguration")]),e._v(" object or a configuration dictionary")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("config")])])]),e._v(" "),a("dd",[e._v("A "),a("code",[e._v("PipelineConfiguration")]),e._v(" object or a configuration dict")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("vocab_path")])])]),e._v(" "),a("dd",[e._v("If provided, the pipeline vocabulary will be loaded from this path")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("pipeline")])]),e._v(" "),a("dd",[e._v("A configured pipeline")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-pretrained"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-pretrained"}},[e._v("#")]),e._v(" from_pretrained "),a("Badge",{attrs:{text:"Static method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("from_pretrained")]),e._v("("),a("span",[e._v("path: Union[str, pathlib.Path]) -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[e._v("Pipeline")])]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Loads a pretrained pipeline providing a "),a("em",[e._v("model.tar.gz")]),e._v(" file path")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("path")])])]),e._v(" "),a("dd",[e._v("The path to the "),a("em",[e._v("model.tar.gz")]),e._v(" file of a pretrained "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[e._v("Pipeline")])])])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("pipeline")])]),e._v(" "),a("dd",[e._v("A pretrained pipeline")])])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"instance-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[e._v("#")]),e._v(" Instance variables")]),e._v("\n")]),e._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.name"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("name")]),e._v(" : str")])]),e._v(" "),a("dd",[a("p",[e._v("Gets the pipeline name")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.inputs"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("inputs")]),e._v(" : List[str]")])]),e._v(" "),a("dd",[a("p",[e._v("Gets the pipeline input field names")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.output"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("output")]),e._v(" : List[str]")])]),e._v(" "),a("dd",[a("p",[e._v("Gets the pipeline output field names")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.backbone"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("backbone")]),e._v(" : "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"backbone.html#biome.text.backbone.ModelBackbone"}},[e._v("ModelBackbone")])])]),e._v(" "),a("dd",[a("p",[e._v("Gets the model backbone of the pipeline")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.head"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("head")]),e._v(" : "),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[e._v("TaskHead")])])]),e._v(" "),a("dd",[a("p",[e._v("Gets the pipeline task head")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.vocab"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("vocab")]),e._v(" : allennlp.data.vocabulary.Vocabulary")])]),e._v(" "),a("dd",[a("p",[e._v("Gets the pipeline vocabulary")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.config"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("config")]),e._v(" : "),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[e._v("PipelineConfiguration")])])]),e._v(" "),a("dd",[a("p",[e._v("Gets the pipeline configuration")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.model"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("model")]),e._v(" : "),a("a",{attrs:{title:"biome.text.model.PipelineModel",href:"model.html#biome.text.model.PipelineModel"}},[e._v("PipelineModel")])])]),e._v(" "),a("dd",[a("p",[e._v("Gets the underlying model")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.type_name"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("type_name")]),e._v(" : str")])]),e._v(" "),a("dd",[a("p",[e._v("The pipeline name. Equivalent to task head name")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.num_trainable_parameters"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("num_trainable_parameters")]),e._v(" : int")])]),e._v(" "),a("dd",[a("p",[e._v("Number of trainable parameters present in the model.")]),e._v(" "),a("p",[e._v("At training time, this number can change when freezing/unfreezing certain parameter groups.")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.num_parameters"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("num_parameters")]),e._v(" : int")])]),e._v(" "),a("dd",[a("p",[e._v("Number of parameters present in the model.")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.named_trainable_parameters"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("named_trainable_parameters")]),e._v(" : List[str]")])]),e._v(" "),a("dd",[a("p",[e._v("Returns the names of the trainable parameters in the pipeline")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.model_path"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("model_path")]),e._v(" : str")])]),e._v(" "),a("dd",[a("p",[e._v("Returns the file path to the serialized version of the last trained model")])])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-logger"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-logger"}},[e._v("#")]),e._v(" init_prediction_logger "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("init_prediction_logger")]),e._v(" ("),e._v("\n  self,\n  output_dir: str,\n  max_logging_size: int = 100,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Initializes the prediction logging.")]),e._v(" "),a("p",[e._v("If initialized, all predictions will be logged to a file called "),a("em",[e._v("predictions.json")]),e._v(" in the "),a("code",[e._v("output_dir")]),e._v(".")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("output_dir")])]),e._v(" : "),a("code",[e._v("str")])]),e._v(" "),a("dd",[e._v("Path to the folder in which we create the "),a("em",[e._v("predictions.json")]),e._v(" file.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("max_logging_size")])]),e._v(" : "),a("code",[e._v("int")])]),e._v(" "),a("dd",[e._v("Max disk size to use for prediction logs")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-cache"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-cache"}},[e._v("#")]),e._v(" init_prediction_cache "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("init_prediction_cache")]),e._v(" ("),e._v("\n  self,\n  max_size: int,\n)  -> NoneType\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Initializes the cache for input predictions")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("max_size")])])]),e._v(" "),a("dd",[e._v("Save up to max_size most recent (inputs).")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"create-vocab"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#create-vocab"}},[e._v("#")]),e._v(" create_vocab "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("create_vocab")]),e._v(" ("),e._v("\n  self,\n  instance_datasets: Iterable[Union["),a("a",{attrs:{title:"biome.text.dataset.AllennlpDataset",href:"dataset.html#biome.text.dataset.AllennlpDataset"}},[e._v("AllennlpDataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpLazyDataset",href:"dataset.html#biome.text.dataset.AllennlpLazyDataset"}},[e._v("AllennlpLazyDataset")]),e._v("]],\n  config: Union["),a("a",{attrs:{title:"biome.text.configuration.VocabularyConfiguration",href:"configuration.html#biome.text.configuration.VocabularyConfiguration"}},[e._v("VocabularyConfiguration")]),e._v(", NoneType] = None,\n)  -> allennlp.data.vocabulary.Vocabulary\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Creates and updates the vocab of the pipeline.")]),e._v(" "),a("p",[e._v("NOTE: The trainer calls this method for you. You can use this method in case you want\nto create the vocab outside of the training process.")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("instance_datasets")])])]),e._v(" "),a("dd",[e._v("A list of instance datasets from which to create the vocabulary.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("config")])])]),e._v(" "),a("dd",[e._v("Configurations for the vocab creation. Default: "),a("code",[e._v("VocabularyConfiguration()")]),e._v(".")])]),e._v(" "),a("h2",{attrs:{id:"examples"}},[e._v("Examples")]),e._v(" "),a("pre",[a("code",{staticClass:"language-python"},[e._v('>>> from biome.text import Pipeline, Dataset\n>>> pl = Pipeline.from_config(\n...     {"name": "example", "head":{"type": "TextClassification", "labels": ["pos", "neg"]}}\n... )\n>>> dataset = Dataset.from_dict({"text": ["Just an example"], "label": ["pos"]})\n>>> instance_dataset = dataset.to_instances(pl)\n>>> vocab = pl.create_vocab([instance_dataset])\n')])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"predict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predict"}},[e._v("#")]),e._v(" predict "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("predict")]),e._v(" ("),e._v("\n  self,\n  *args,\n  batch: Union[List[Dict[str, Any]], NoneType] = None,\n  add_tokens: bool = False,\n  add_attributions: bool = False,\n  attributions_kwargs: Union[Dict, NoneType] = None,\n  **kwargs,\n)  -> Union[Dict[str, numpy.ndarray], List[Union[Dict[str, numpy.ndarray], NoneType]]]\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Returns a prediction given some input data based on the current state of the model")]),e._v(" "),a("p",[e._v("The accepted input is dynamically calculated and can be checked via the "),a("code",[e._v("self.inputs")]),e._v(" attribute\n("),a("code",[e._v("print("),a("a",{attrs:{title:"biome.text.pipeline.Pipeline.inputs",href:"#biome.text.pipeline.Pipeline.inputs"}},[e._v("Pipeline.inputs")]),e._v(")")]),e._v(")")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("em",[e._v("args/")]),e._v("*kwargs")]),e._v(" "),a("dt",[e._v("These are dynamically updated and correspond to the pipeline's "),a("code",[e._v("self.inputs")]),e._v(".")]),e._v(" "),a("dt",[e._v("If provided, the "),a("code",[e._v("batch")]),e._v(" parameter will be ignored.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("batch")])])]),e._v(" "),a("dd",[e._v("A list of dictionaries that represents a batch of inputs. The dictionary keys must comply with the\n"),a("code",[e._v("self.inputs")]),e._v(" attribute. Predicting batches should typically be faster than repeated calls with args/kwargs.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("add_tokens")])])]),e._v(" "),a("dd",[e._v("If true, adds a 'tokens' key in the prediction that contains the tokenized input.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("add_attributions")])])]),e._v(" "),a("dd",[e._v("If true, adds a 'attributions' key that contains attributions of the input to the prediction.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("attributions_kwargs")])])]),e._v(" "),a("dd",[e._v("This dict is directly passed on to the "),a("code",[e._v("TaskHead.compute_attributions()")]),e._v(".")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("predictions")])]),e._v(" "),a("dd",[e._v("A dictionary or a list of dictionaries containing the predictions and additional information.\nIf a prediction fails for a single input in the batch, its return value will be "),a("code",[e._v("None")]),e._v(".")])]),e._v(" "),a("h2",{attrs:{id:"raises"}},[e._v("Raises")]),e._v(" "),a("dl",[a("dt",[a("code",[a("a",{attrs:{title:"biome.text.pipeline.PredictionError",href:"#biome.text.pipeline.PredictionError"}},[e._v("PredictionError")])])]),e._v(" "),a("dd",[e._v("Failed to predict the single input or the whole batch")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"evaluate"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#evaluate"}},[e._v("#")]),e._v(" evaluate "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("evaluate")]),e._v(" ("),e._v("\n  self,\n  test_dataset: Union["),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[e._v("Dataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpDataset",href:"dataset.html#biome.text.dataset.AllennlpDataset"}},[e._v("AllennlpDataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpLazyDataset",href:"dataset.html#biome.text.dataset.AllennlpLazyDataset"}},[e._v("AllennlpLazyDataset")]),e._v("],\n  batch_size: int = 16,\n  lazy: bool = False,\n  output_dir: Union[pathlib.Path, str, NoneType] = None,\n  verbose: bool = True,\n)  -> Dict[str, Any]\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Evaluate your model on a test dataset")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("test_dataset")])])]),e._v(" "),a("dd",[e._v("The test data set.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("batch_size")])])]),e._v(" "),a("dd",[e._v("The batch size. Default: 16.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("lazy")])])]),e._v(" "),a("dd",[e._v("If True, instances are lazily loaded from disk, otherwise they are loaded into memory.\nIgnored when "),a("code",[e._v("test_dataset")]),e._v(" is a "),a("code",[e._v("InstanceDataset")]),e._v(". Default: False.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("output_dir")])])]),e._v(" "),a("dd",[e._v("Save a "),a("code",[e._v("metrics.json")]),e._v(" to this output directory. Default: None.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("verbose")])])]),e._v(" "),a("dd",[e._v("If True, prints the test results. Default: True.")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("Dict[str, Any]")])]),e._v(" "),a("dd",[e._v("A dictionary with the metrics")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"set-head"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#set-head"}},[e._v("#")]),e._v(" set_head "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("set_head")]),e._v(" ("),e._v("\n  self,\n  type: Type["),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[e._v("TaskHead")]),e._v("],\n  **kwargs,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Sets a new task head for the pipeline")]),e._v(" "),a("p",[e._v("Call this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("type")])]),e._v(" : "),a("code",[e._v("Type[TaskHead]")])]),e._v(" "),a("dd",[e._v("The "),a("code",[e._v("TaskHead")]),e._v(" class to be set for the pipeline (e.g., "),a("code",[e._v("TextClassification")])])]),e._v(" "),a("p",[e._v("**kwargs:\nThe "),a("code",[e._v("TaskHead")]),e._v(" specific arguments (e.g., the classification head needs a "),a("code",[e._v("pooler")]),e._v(" layer)")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"model-parameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#model-parameters"}},[e._v("#")]),e._v(" model_parameters "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("model_parameters")]),e._v("("),a("span",[e._v("self) -> Iterator[Tuple[str, torch.Tensor]]")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Returns an iterator over all model parameters, yielding the name and the parameter itself.")]),e._v(" "),a("h2",{attrs:{id:"examples"}},[e._v("Examples")]),e._v(" "),a("p",[e._v("You can use this to freeze certain parameters in the training:")]),e._v(" "),a("pre",[a("code",{staticClass:"language-python"},[e._v('>>> pipeline = Pipeline.from_config({\n...     "name": "model_parameters_example",\n...     "head": {"type": "TextClassification", "labels": ["a", "b"]},\n... })\n>>> for name, parameter in pipeline.model_parameters():\n...     if not name.endswith("bias"):\n...         parameter.requires_grad = False\n')])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"copy"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#copy"}},[e._v("#")]),e._v(" copy "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("copy")]),e._v("("),a("span",[e._v("self) -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[e._v("Pipeline")])]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Returns a copy of the pipeline")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"save"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#save"}},[e._v("#")]),e._v(" save "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("save")]),e._v(" ("),e._v("\n  self,\n  directory: Union[str, pathlib.Path],\n)  -> str\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Saves the pipeline in the given directory as "),a("code",[e._v("model.tar.gz")]),e._v(" file.")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("directory")])])]),e._v(" "),a("dd",[e._v("Save the 'model.tar.gz' file to this directory.")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("file_path")])]),e._v(" "),a("dd",[e._v("Path to the 'model.tar.gz' file.")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"to-mlflow"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#to-mlflow"}},[e._v("#")]),e._v(" to_mlflow "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("to_mlflow")]),e._v(" ("),e._v("\n  self,\n  tracking_uri: Union[str, NoneType] = None,\n  experiment_id: Union[int, NoneType] = None,\n  run_name: str = 'log_biometext_model',\n  input_example: Union[Dict, NoneType] = None,\n  conda_env: Union[Dict, NoneType] = None,\n)  -> str\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Logs the pipeline as MLFlow Model to a MLFlow Tracking server")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("tracking_uri")])])]),e._v(" "),a("dd",[e._v("The URI of the MLFlow tracking server, MLFlow defaults to './mlruns'. Default: None")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("experiment_id")])])]),e._v(" "),a("dd",[e._v("ID of the experiment under which to create the logging run. If this argument is unspecified,\nwill look for valid experiment in the following order: activated using "),a("code",[e._v("mlflow.set_experiment")]),e._v(",\n"),a("code",[e._v("MLFLOW_EXPERIMENT_NAME")]),e._v(" environment variable, "),a("code",[e._v("MLFLOW_EXPERIMENT_ID")]),e._v(" environment variable,\nor the default experiment as defined by the tracking server.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("run_name")])])]),e._v(" "),a("dd",[e._v("The name of the MLFlow run logging the model. Default: 'log_biometext_model'.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("input_example")])])]),e._v(" "),a("dd",[e._v("You can provide an input example in the form of a dictionary. For example, for a TextClassification head\nthis would be "),a("code",[e._v('{"text": "This is an input example"}')]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("conda_env")])])]),e._v(" "),a("dd",[e._v("This conda environment is used when serving the model via "),a("code",[e._v("mlflow models serve")]),e._v('. Default:\nconda_env = {\n"name": "mlflow-dev",\n"channels": ["defaults", "conda-forge"],\n"dependencies": ["python=3.7.9", "pip>=20.3.0", {"pip": ["biome-text=={'),a("strong",[e._v("version")]),e._v('}"]}],\n}')])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("model_uri")])]),e._v(" "),a("dd",[e._v("The URI of the logged MLFlow model. The model gets logged as an artifact to the corresponding run.")])]),e._v(" "),a("h2",{attrs:{id:"examples"}},[e._v("Examples")]),e._v(" "),a("p",[e._v("After logging the pipeline to MLFlow you can use the MLFlow model for inference:")]),e._v(" "),a("pre",[a("code",{staticClass:"language-python"},[e._v('>>> import mlflow, pandas, biome.text\n>>> pipeline = biome.text.Pipeline.from_config({\n...     "name": "to_mlflow_example",\n...     "head": {"type": "TextClassification", "labels": ["a", "b"]},\n... })\n>>> model_uri = pipeline.to_mlflow()\n>>> model = mlflow.pyfunc.load_model(model_uri)\n>>> prediction: pandas.DataFrame = model.predict(pandas.DataFrame([{"text": "Test this text"}]))\n')])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"predictionerror"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predictionerror"}},[e._v("#")]),e._v(" PredictionError "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("PredictionError")]),e._v(" (...)"),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Exception for a failed prediction of a single input or a whole batch")]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[e._v("builtins.Exception")]),e._v(" "),a("li",[e._v("builtins.BaseException")])])])}),[],!1,null,null,null);t.default=s.exports}}]);