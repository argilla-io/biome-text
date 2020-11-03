(window.webpackJsonp=window.webpackJsonp||[]).push([[55],{430:function(t,e,a){"use strict";a.r(e);var i=a(26),s=Object(i.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-pipeline"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-pipeline"}},[t._v("#")]),t._v(" biome.text.pipeline "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"pipeline"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pipeline"}},[t._v("#")]),t._v(" Pipeline "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("Pipeline")]),t._v(" ()"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Manages NLP models configuration and actions.")]),t._v(" "),a("p",[t._v("Use "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])]),t._v(" for creating new models from a configuration or loading a pre-trained model.")]),t._v(" "),a("p",[t._v("Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"subclasses"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#subclasses"}},[t._v("#")]),t._v(" Subclasses")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[t._v("biome.text.pipeline._BlankPipeline")]),t._v(" "),a("li",[t._v("biome.text.pipeline._PipelineCopy")]),t._v(" "),a("li",[t._v("biome.text.pipeline._PreTrainedPipeline")])]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-yaml"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-yaml"}},[t._v("#")]),t._v(" from_yaml "),a("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("from_yaml")]),t._v(" ("),t._v("\n  path: str,\n  vocab_path: Union[str, NoneType] = None,\n)  -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Creates a pipeline from a config yaml file")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("path")])]),t._v(" : "),a("code",[t._v("str")])]),t._v(" "),a("dd",[t._v("The path to a YAML configuration file")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("vocab_path")])]),t._v(" : "),a("code",[t._v("Optional[str]")])]),t._v(" "),a("dd",[t._v("If provided, the pipeline vocab will be loaded from this path")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("pipeline")])]),t._v(" : "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])])]),t._v(" "),a("dd",[t._v("A configured pipeline")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-config"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-config"}},[t._v("#")]),t._v(" from_config "),a("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("from_config")]),t._v(" ("),t._v("\n  config: Union["),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[t._v("PipelineConfiguration")]),t._v(", dict],\n  vocab_path: Union[str, NoneType] = None,\n)  -> biome.text.pipeline._BlankPipeline\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Creates a pipeline from a "),a("code",[t._v("PipelineConfiguration")]),t._v(" object or a configuration dictionary")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("config")])]),t._v(" : "),a("code",[t._v("Union[PipelineConfiguration, dict]")])]),t._v(" "),a("dd",[t._v("A "),a("code",[t._v("PipelineConfiguration")]),t._v(" object or a configuration dict")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("vocab_path")])]),t._v(" : "),a("code",[t._v("Optional[str]")])]),t._v(" "),a("dd",[t._v("If provided, the pipeline vocabulary will be loaded from this path")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("pipeline")])]),t._v(" : "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])])]),t._v(" "),a("dd",[t._v("A configured pipeline")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-pretrained"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-pretrained"}},[t._v("#")]),t._v(" from_pretrained "),a("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("from_pretrained")]),t._v(" ("),t._v("\n  path: str,\n  **kwargs,\n)  -> biome.text.pipeline._PreTrainedPipeline\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Loads a pipeline from a pre-trained pipeline providing a "),a("em",[t._v("model.tar.gz")]),t._v(" file path")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("path")])]),t._v(" : "),a("code",[t._v("str")])]),t._v(" "),a("dd",[t._v("The path to the "),a("em",[t._v("model.tar.gz")]),t._v(" file of a pre-trained "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])])])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("pipeline")])]),t._v(" : "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])])]),t._v(" "),a("dd",[t._v("A configured pipeline")])])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"instance-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[t._v("#")]),t._v(" Instance variables")]),t._v("\n")]),t._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.name"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("name")]),t._v(" : str")])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline name")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.inputs"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("inputs")]),t._v(" : List[str]")])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline input field names")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.output"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("output")]),t._v(" : List[str]")])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline output field names")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.backbone"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("backbone")]),t._v(" : "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"backbone.html#biome.text.backbone.ModelBackbone"}},[t._v("ModelBackbone")])])]),t._v(" "),a("dd",[a("p",[t._v("Gets the model backbone of the pipeline")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.head"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("head")]),t._v(" : "),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")])])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline task head")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.config"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("config")]),t._v(" : "),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[t._v("PipelineConfiguration")])])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline configuration")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.type_name"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("type_name")]),t._v(" : str")])]),t._v(" "),a("dd",[a("p",[t._v("The pipeline name. Equivalent to task head name")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.num_trainable_parameters"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("num_trainable_parameters")]),t._v(" : int")])]),t._v(" "),a("dd",[a("p",[t._v("Number of trainable parameters present in the model.")]),t._v(" "),a("p",[t._v("At training time, this number can change when freezing/unfreezing certain parameter groups.")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.num_parameters"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("num_parameters")]),t._v(" : int")])]),t._v(" "),a("dd",[a("p",[t._v("Number of parameters present in the model.")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.named_trainable_parameters"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("named_trainable_parameters")]),t._v(" : List[str]")])]),t._v(" "),a("dd",[a("p",[t._v("Returns the names of the trainable parameters in the pipeline")])])]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-logger"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-logger"}},[t._v("#")]),t._v(" init_prediction_logger "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("init_prediction_logger")]),t._v(" ("),t._v("\n  self,\n  output_dir: str,\n  max_logging_size: int = 100,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Initializes the prediction logging.")]),t._v(" "),a("p",[t._v("If initialized, all predictions will be logged to a file called "),a("em",[t._v("predictions.json")]),t._v(" in the "),a("code",[t._v("output_dir")]),t._v(".")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("output_dir")])]),t._v(" : "),a("code",[t._v("str")])]),t._v(" "),a("dd",[t._v("Path to the folder in which we create the "),a("em",[t._v("predictions.json")]),t._v(" file.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("max_logging_size")])]),t._v(" : "),a("code",[t._v("int")])]),t._v(" "),a("dd",[t._v("Max disk size to use for prediction logs")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-cache"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-cache"}},[t._v("#")]),t._v(" init_prediction_cache "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("init_prediction_cache")]),t._v(" ("),t._v("\n  self,\n  max_size: int,\n)  -> NoneType\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Initializes the cache for input predictions")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("max_size")])])]),t._v(" "),a("dd",[t._v("Save up to max_size most recent (inputs).")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"find-lr"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#find-lr"}},[t._v("#")]),t._v(" find_lr "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("find_lr")]),t._v(" ("),t._v("\n  self,\n  trainer_config: "),a("a",{attrs:{title:"biome.text.configuration.TrainerConfiguration",href:"configuration.html#biome.text.configuration.TrainerConfiguration"}},[t._v("TrainerConfiguration")]),t._v(",\n  find_lr_config: "),a("a",{attrs:{title:"biome.text.configuration.FindLRConfiguration",href:"configuration.html#biome.text.configuration.FindLRConfiguration"}},[t._v("FindLRConfiguration")]),t._v(",\n  training_data: Union["),a("a",{attrs:{title:"biome.text.data.datasource.DataSource",href:"data/datasource.html#biome.text.data.datasource.DataSource"}},[t._v("DataSource")]),t._v(", "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(", allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset],\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns a learning rate scan on the model.")]),t._v(" "),a("p",[t._v("It increases the learning rate step by step while recording the losses.\nFor a guide on how to select the learning rate please refer to this excellent\n"),a("a",{attrs:{href:"https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0"}},[t._v("blog post")])]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("trainer_config")])])]),t._v(" "),a("dd",[t._v("A trainer configuration")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("find_lr_config")])])]),t._v(" "),a("dd",[t._v("A configuration for finding the learning rate")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("training_data")])])]),t._v(" "),a("dd",[t._v("The training data")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("p",[t._v("(learning_rates, losses)\nReturns a list of learning rates and corresponding losses.\nNote: The losses are recorded before applying the corresponding learning rate")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"train"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#train"}},[t._v("#")]),t._v(" train "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("train")]),t._v(" ("),t._v("\n  self,\n  output: str,\n  training: Union["),a("a",{attrs:{title:"biome.text.data.datasource.DataSource",href:"data/datasource.html#biome.text.data.datasource.DataSource"}},[t._v("DataSource")]),t._v(", "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(", allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset],\n  trainer: Union["),a("a",{attrs:{title:"biome.text.configuration.TrainerConfiguration",href:"configuration.html#biome.text.configuration.TrainerConfiguration"}},[t._v("TrainerConfiguration")]),t._v(", NoneType] = None,\n  validation: Union["),a("a",{attrs:{title:"biome.text.data.datasource.DataSource",href:"data/datasource.html#biome.text.data.datasource.DataSource"}},[t._v("DataSource")]),t._v(", "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(", allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset, NoneType] = None,\n  test: Union["),a("a",{attrs:{title:"biome.text.data.datasource.DataSource",href:"data/datasource.html#biome.text.data.datasource.DataSource"}},[t._v("DataSource")]),t._v(", "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(", allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset, NoneType] = None,\n  extend_vocab: Union["),a("a",{attrs:{title:"biome.text.configuration.VocabularyConfiguration",href:"configuration.html#biome.text.configuration.VocabularyConfiguration"}},[t._v("VocabularyConfiguration")]),t._v(", NoneType] = None,\n  loggers: List["),a("a",{attrs:{title:"biome.text.loggers.BaseTrainLogger",href:"loggers.html#biome.text.loggers.BaseTrainLogger"}},[t._v("BaseTrainLogger")]),t._v("] = None,\n  lazy: bool = True,\n  restore: bool = False,\n  quiet: bool = False,\n)  -> "),a("a",{attrs:{title:"biome.text.training_results.TrainingResults",href:"training_results.html#biome.text.training_results.TrainingResults"}},[t._v("TrainingResults")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Launches a training run with the specified configurations and data sources")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("output")])])]),t._v(" "),a("dd",[t._v("The experiment output path")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("training")])])]),t._v(" "),a("dd",[t._v("The training DataSource")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("trainer")])])]),t._v(" "),a("dd",[t._v("The trainer file path")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("validation")])])]),t._v(" "),a("dd",[t._v("The validation DataSource (optional)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("test")])])]),t._v(" "),a("dd",[t._v("The test DataSource (optional)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("extend_vocab")])])]),t._v(" "),a("dd",[t._v("Extends the vocabulary tokens with the provided VocabularyConfiguration")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("loggers")])])]),t._v(" "),a("dd",[t._v("A list of loggers that execute a callback before the training, after each epoch,\nand at the end of the training (see "),a("code",[t._v("biome.text.logger.MlflowLogger")]),t._v(", for example)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("lazy")])])]),t._v(" "),a("dd",[t._v("If true, load the data lazily from disk, otherwise load them in memory.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("restore")])])]),t._v(" "),a("dd",[t._v("If enabled, tries to read previous training status from the "),a("code",[t._v("output")]),t._v(" folder and\ncontinues the training process")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("quiet")])])]),t._v(" "),a("dd",[t._v("If enabled, disables most logging messages keeping only warning and error messages.\nIn any case, all logging info will be stored into a file at ${output}/train.log")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("training_results")])]),t._v(" "),a("dd",[t._v("Training results including the generated model path and the related metrics")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"create-dataset"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#create-dataset"}},[t._v("#")]),t._v(" create_dataset "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("create_dataset")]),t._v(" ("),t._v("\n  self,\n  datasource: "),a("a",{attrs:{title:"biome.text.data.datasource.DataSource",href:"data/datasource.html#biome.text.data.datasource.DataSource"}},[t._v("DataSource")]),t._v(",\n  lazy: bool = False,\n)  -> Union[allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Creates an instances torch Dataset from an data source")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("p",[t._v("datasource:\nThe source of data\nlazy:\nIf enabled, the returned dataset is a subclass of "),a("code",[t._v("torch.data.utils.IterableDataset")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("A torch Dataset containing the instances collection")])]),t._v(" "),a("dd",[t._v(" ")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"predict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predict"}},[t._v("#")]),t._v(" predict "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("predict")]),t._v(" ("),t._v("\n  self,\n  *args,\n  **kwargs,\n)  -> Dict[str, numpy.ndarray]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns a prediction given some input data based on the current state of the model")]),t._v(" "),a("p",[t._v("The accepted input is dynamically calculated and can be checked via the "),a("code",[t._v("self.inputs")]),t._v(" attribute\n("),a("code",[t._v("print("),a("a",{attrs:{title:"biome.text.pipeline.Pipeline.inputs",href:"#biome.text.pipeline.Pipeline.inputs"}},[t._v("Pipeline.inputs")]),t._v(")")]),t._v(")")]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("predictions")])]),t._v(" : "),a("code",[t._v("Dict[str, numpy.ndarray]")])]),t._v(" "),a("dd",[t._v("A dictionary containing the predictions and additional information")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"predict-batch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predict-batch"}},[t._v("#")]),t._v(" predict_batch "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("predict_batch")]),t._v(" ("),t._v("\n  self,\n  input_dicts: Iterable[Dict[str, Any]],\n)  -> List[Dict[str, numpy.ndarray]]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns predictions given some input data based on the current state of the model")]),t._v(" "),a("p",[t._v("The predictions will be computed batch-wise, which is faster\nthan calling "),a("code",[t._v("self.predict")]),t._v(" for every single input data.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("input_dicts")])])]),t._v(" "),a("dd",[t._v("The input data. The keys of the dicts must comply with the "),a("code",[t._v("self.inputs")]),t._v(" attribute")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"explain"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#explain"}},[t._v("#")]),t._v(" explain "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("explain")]),t._v(" ("),t._v("\n  self,\n  *args,\n  n_steps: int = 5,\n  **kwargs,\n)  -> Dict[str, Any]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns a prediction given some input data including the attribution of each token to the prediction.")]),t._v(" "),a("p",[t._v("The attributions are calculated by means of the "),a("a",{attrs:{href:"https://arxiv.org/abs/1703.01365"}},[t._v("Integrated Gradients")]),t._v(" method.")]),t._v(" "),a("p",[t._v("The accepted input is dynamically calculated and can be checked via the "),a("code",[t._v("self.inputs")]),t._v(" attribute\n("),a("code",[t._v("print("),a("a",{attrs:{title:"biome.text.pipeline.Pipeline.inputs",href:"#biome.text.pipeline.Pipeline.inputs"}},[t._v("Pipeline.inputs")]),t._v(")")]),t._v(")")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("n_steps")])]),t._v(" : "),a("code",[t._v("int")])]),t._v(" "),a("dd",[t._v("The number of steps used when calculating the attribution of each token.\nIf the number of steps is less than 1, the attributions will not be calculated.")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("predictions")])]),t._v(" : "),a("code",[t._v("Dict[str, numpy.ndarray]")])]),t._v(" "),a("dd",[t._v("A dictionary containing the predictions and attributions")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"explain-batch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#explain-batch"}},[t._v("#")]),t._v(" explain_batch "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("explain_batch")]),t._v(" ("),t._v("\n  self,\n  input_dicts: Iterable[Dict[str, Any]],\n  n_steps: int = 5,\n)  -> List[Dict[str, numpy.ndarray]]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns a prediction given some input data including the attribution of each token to the prediction.")]),t._v(" "),a("p",[t._v("The predictions will be computed batch-wise, which is faster\nthan calling "),a("code",[t._v("self.predict")]),t._v(" for every single input data.")]),t._v(" "),a("p",[t._v("The attributions are calculated by means of the "),a("a",{attrs:{href:"https://arxiv.org/abs/1703.01365"}},[t._v("Integrated Gradients")]),t._v(" method.")]),t._v(" "),a("p",[t._v("The accepted input is dynamically calculated and can be checked via the "),a("code",[t._v("self.inputs")]),t._v(" attribute\n("),a("code",[t._v("print("),a("a",{attrs:{title:"biome.text.pipeline.Pipeline.inputs",href:"#biome.text.pipeline.Pipeline.inputs"}},[t._v("Pipeline.inputs")]),t._v(")")]),t._v(")")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("input_dicts")])])]),t._v(" "),a("dd",[t._v("The input data. The keys of the dicts must comply with the "),a("code",[t._v("self.inputs")]),t._v(" attribute")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("n_steps")])])]),t._v(" "),a("dd",[t._v("The number of steps used when calculating the attribution of each token.\nIf the number of steps is less than 1, the attributions will not be calculated.")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("predictions")])]),t._v(" "),a("dd",[t._v("A list of dictionaries containing the predictions and attributions")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"save-vocabulary"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#save-vocabulary"}},[t._v("#")]),t._v(" save_vocabulary "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("save_vocabulary")]),t._v(" ("),t._v("\n  self,\n  directory: str,\n)  -> NoneType\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Saves the pipeline's vocabulary in a directory")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("directory")])]),t._v(" : "),a("code",[t._v("str")])]),t._v(" "),a("dd",[t._v(" ")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"serve"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#serve"}},[t._v("#")]),t._v(" serve "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("serve")]),t._v(" ("),t._v("\n  self,\n  port: int = 9998,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Launches a REST prediction service with the current model")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("port")])]),t._v(" : "),a("code",[t._v("int")])]),t._v(" "),a("dd",[t._v("The port on which the prediction service will be running (default: 9998)")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"set-head"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#set-head"}},[t._v("#")]),t._v(" set_head "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("set_head")]),t._v(" ("),t._v("\n  self,\n  type: Type["),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")]),t._v("],\n  **kwargs,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Sets a new task head for the pipeline")]),t._v(" "),a("p",[t._v("Call this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("type")])]),t._v(" : "),a("code",[t._v("Type[TaskHead]")])]),t._v(" "),a("dd",[t._v("The "),a("code",[t._v("TaskHead")]),t._v(" class to be set for the pipeline (e.g., "),a("code",[t._v("TextClassification")])])]),t._v(" "),a("p",[t._v("**kwargs:\nThe "),a("code",[t._v("TaskHead")]),t._v(" specific arguments (e.g., the classification head needs a "),a("code",[t._v("pooler")]),t._v(" layer)")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"has-empty-vocab"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#has-empty-vocab"}},[t._v("#")]),t._v(" has_empty_vocab "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("has_empty_vocab")]),t._v("("),a("span",[t._v("self) -> bool")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Determines if a pipeline has an empty vocab under configured features")])])])])}),[],!1,null,null,null);e.default=s.exports}}]);