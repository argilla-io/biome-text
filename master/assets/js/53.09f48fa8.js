(window.webpackJsonp=window.webpackJsonp||[]).push([[53],{478:function(t,e,a){"use strict";a.r(e);var i=a(26),n=Object(i.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-pipeline"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-pipeline"}},[t._v("#")]),t._v(" biome.text.pipeline "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"pipeline"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pipeline"}},[t._v("#")]),t._v(" Pipeline "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("Pipeline")]),t._v(" (model: "),a("a",{attrs:{title:"biome.text.model.PipelineModel",href:"model.html#biome.text.model.PipelineModel"}},[t._v("PipelineModel")]),t._v(", config: "),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[t._v("PipelineConfiguration")]),t._v(")"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Manages NLP models configuration and actions.")]),t._v(" "),a("p",[t._v("Use "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])]),t._v(" for creating new models from a configuration or loading a pretrained model.")]),t._v(" "),a("p",[t._v("Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.")]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-yaml"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-yaml"}},[t._v("#")]),t._v(" from_yaml "),a("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("from_yaml")]),t._v(" ("),t._v("\n  path: str,\n  vocab_path: Union[str, NoneType] = None,\n)  -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Creates a pipeline from a config yaml file")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("path")])])]),t._v(" "),a("dd",[t._v("The path to a YAML configuration file")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("vocab_path")])])]),t._v(" "),a("dd",[t._v("If provided, the pipeline vocab will be loaded from this path")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("pipeline")])]),t._v(" "),a("dd",[t._v("A configured pipeline")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-config"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-config"}},[t._v("#")]),t._v(" from_config "),a("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("from_config")]),t._v(" ("),t._v("\n  config: Union["),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[t._v("PipelineConfiguration")]),t._v(", dict],\n  vocab_path: Union[str, NoneType] = None,\n)  -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Creates a pipeline from a "),a("code",[t._v("PipelineConfiguration")]),t._v(" object or a configuration dictionary")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("config")])])]),t._v(" "),a("dd",[t._v("A "),a("code",[t._v("PipelineConfiguration")]),t._v(" object or a configuration dict")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("vocab_path")])])]),t._v(" "),a("dd",[t._v("If provided, the pipeline vocabulary will be loaded from this path")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("pipeline")])]),t._v(" "),a("dd",[t._v("A configured pipeline")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-pretrained"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-pretrained"}},[t._v("#")]),t._v(" from_pretrained "),a("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("from_pretrained")]),t._v("("),a("span",[t._v("path: Union[str, pathlib.Path]) -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Loads a pretrained pipeline providing a "),a("em",[t._v("model.tar.gz")]),t._v(" file path")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("path")])])]),t._v(" "),a("dd",[t._v("The path to the "),a("em",[t._v("model.tar.gz")]),t._v(" file of a pretrained "),a("code",[a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])])])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("pipeline")])]),t._v(" "),a("dd",[t._v("A pretrained pipeline")])])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"instance-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[t._v("#")]),t._v(" Instance variables")]),t._v("\n")]),t._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.name"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("name")]),t._v(" : str")])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline name")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.inputs"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("inputs")]),t._v(" : List[str]")])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline input field names")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.output"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("output")]),t._v(" : List[str]")])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline output field names")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.backbone"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("backbone")]),t._v(" : "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"backbone.html#biome.text.backbone.ModelBackbone"}},[t._v("ModelBackbone")])])]),t._v(" "),a("dd",[a("p",[t._v("Gets the model backbone of the pipeline")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.head"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("head")]),t._v(" : "),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")])])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline task head")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.vocab"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("vocab")]),t._v(" : allennlp.data.vocabulary.Vocabulary")])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline vocabulary")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.config"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("config")]),t._v(" : "),a("a",{attrs:{title:"biome.text.configuration.PipelineConfiguration",href:"configuration.html#biome.text.configuration.PipelineConfiguration"}},[t._v("PipelineConfiguration")])])]),t._v(" "),a("dd",[a("p",[t._v("Gets the pipeline configuration")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.model"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("model")]),t._v(" : "),a("a",{attrs:{title:"biome.text.model.PipelineModel",href:"model.html#biome.text.model.PipelineModel"}},[t._v("PipelineModel")])])]),t._v(" "),a("dd",[a("p",[t._v("Gets the underlying model")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.type_name"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("type_name")]),t._v(" : str")])]),t._v(" "),a("dd",[a("p",[t._v("The pipeline name. Equivalent to task head name")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.num_trainable_parameters"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("num_trainable_parameters")]),t._v(" : int")])]),t._v(" "),a("dd",[a("p",[t._v("Number of trainable parameters present in the model.")]),t._v(" "),a("p",[t._v("At training time, this number can change when freezing/unfreezing certain parameter groups.")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.num_parameters"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("num_parameters")]),t._v(" : int")])]),t._v(" "),a("dd",[a("p",[t._v("Number of parameters present in the model.")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.named_trainable_parameters"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("named_trainable_parameters")]),t._v(" : List[str]")])]),t._v(" "),a("dd",[a("p",[t._v("Returns the names of the trainable parameters in the pipeline")])]),t._v(" "),a("dt",{attrs:{id:"biome.text.pipeline.Pipeline.model_path"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("model_path")]),t._v(" : str")])]),t._v(" "),a("dd",[a("p",[t._v("Returns the file path to the serialized version of the last trained model")])])]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-logger"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-logger"}},[t._v("#")]),t._v(" init_prediction_logger "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("init_prediction_logger")]),t._v(" ("),t._v("\n  self,\n  output_dir: str,\n  max_logging_size: int = 100,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Initializes the prediction logging.")]),t._v(" "),a("p",[t._v("If initialized, all predictions will be logged to a file called "),a("em",[t._v("predictions.json")]),t._v(" in the "),a("code",[t._v("output_dir")]),t._v(".")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("output_dir")])]),t._v(" : "),a("code",[t._v("str")])]),t._v(" "),a("dd",[t._v("Path to the folder in which we create the "),a("em",[t._v("predictions.json")]),t._v(" file.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("max_logging_size")])]),t._v(" : "),a("code",[t._v("int")])]),t._v(" "),a("dd",[t._v("Max disk size to use for prediction logs")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-cache"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-cache"}},[t._v("#")]),t._v(" init_prediction_cache "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("init_prediction_cache")]),t._v(" ("),t._v("\n  self,\n  max_size: int,\n)  -> NoneType\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Initializes the cache for input predictions")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("max_size")])])]),t._v(" "),a("dd",[t._v("Save up to max_size most recent (inputs).")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"find-lr"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#find-lr"}},[t._v("#")]),t._v(" find_lr "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("find_lr")]),t._v(" ("),t._v("\n  self,\n  trainer_config: "),a("a",{attrs:{title:"biome.text.configuration.TrainerConfiguration",href:"configuration.html#biome.text.configuration.TrainerConfiguration"}},[t._v("TrainerConfiguration")]),t._v(",\n  find_lr_config: "),a("a",{attrs:{title:"biome.text.configuration.FindLRConfiguration",href:"configuration.html#biome.text.configuration.FindLRConfiguration"}},[t._v("FindLRConfiguration")]),t._v(",\n  training_data: "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(",\n  vocab_config: Union["),a("a",{attrs:{title:"biome.text.configuration.VocabularyConfiguration",href:"configuration.html#biome.text.configuration.VocabularyConfiguration"}},[t._v("VocabularyConfiguration")]),t._v(", str, NoneType] = 'default',\n  lazy: bool = False,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns a learning rate scan on the model.")]),t._v(" "),a("p",[t._v("It increases the learning rate step by step while recording the losses.\nFor a guide on how to select the learning rate please refer to this excellent\n"),a("a",{attrs:{href:"https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0"}},[t._v("blog post")])]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("trainer_config")])])]),t._v(" "),a("dd",[t._v("A trainer configuration")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("find_lr_config")])])]),t._v(" "),a("dd",[t._v("A configuration for finding the learning rate")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("training_data")])])]),t._v(" "),a("dd",[t._v("The training data")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("vocab_config")])])]),t._v(" "),a("dd",[t._v("A "),a("code",[t._v("VocabularyConfiguration")]),t._v(" to create/extend the pipeline's vocabulary.\nIf 'default' (str), we will use the default configuration "),a("code",[t._v("VocabularyConfiguration()")]),t._v(".\nIf None, we will leave the pipeline's vocabulary untouched. Default: 'default'.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("lazy")])])]),t._v(" "),a("dd",[t._v("If true, dataset instances are lazily loaded from disk, otherwise they are loaded and kept in memory.")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("p",[t._v("(learning_rates, losses)\nReturns a list of learning rates and corresponding losses.\nNote: The losses are recorded before applying the corresponding learning rate")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"train"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#train"}},[t._v("#")]),t._v(" train "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("train")]),t._v(" ("),t._v("\n  self,\n  output: str,\n  training: "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(",\n  trainer: Union["),a("a",{attrs:{title:"biome.text.configuration.TrainerConfiguration",href:"configuration.html#biome.text.configuration.TrainerConfiguration"}},[t._v("TrainerConfiguration")]),t._v(", NoneType] = None,\n  validation: Union["),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(", NoneType] = None,\n  test: Union["),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(", NoneType] = None,\n  vocab_config: Union["),a("a",{attrs:{title:"biome.text.configuration.VocabularyConfiguration",href:"configuration.html#biome.text.configuration.VocabularyConfiguration"}},[t._v("VocabularyConfiguration")]),t._v(", str, NoneType] = 'default',\n  loggers: List["),a("a",{attrs:{title:"biome.text.loggers.BaseTrainLogger",href:"loggers.html#biome.text.loggers.BaseTrainLogger"}},[t._v("BaseTrainLogger")]),t._v("] = None,\n  lazy: bool = False,\n  restore: bool = False,\n  quiet: bool = False,\n)  -> "),a("a",{attrs:{title:"biome.text.training_results.TrainingResults",href:"training_results.html#biome.text.training_results.TrainingResults"}},[t._v("TrainingResults")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Launches a training run with the specified configurations and data sources")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("output")])])]),t._v(" "),a("dd",[t._v("The experiment output path")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("training")])])]),t._v(" "),a("dd",[t._v("The training Dataset")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("trainer")])])]),t._v(" "),a("dd",[t._v("The trainer file path")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("validation")])])]),t._v(" "),a("dd",[t._v("The validation Dataset (optional)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("test")])])]),t._v(" "),a("dd",[t._v("The test Dataset (optional)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("vocab_config")])])]),t._v(" "),a("dd",[t._v("A "),a("code",[t._v("VocabularyConfiguration")]),t._v(" to create/extend the pipeline's vocabulary.\nIf 'default' (str), we will use the default configuration "),a("code",[t._v("VocabularyConfiguration()")]),t._v(".\nIf None, we will leave the pipeline's vocabulary untouched. Default: 'default'.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("loggers")])])]),t._v(" "),a("dd",[t._v("A list of loggers that execute a callback before the training, after each epoch,\nand at the end of the training (see "),a("code",[t._v("biome.text.logger.MlflowLogger")]),t._v(", for example)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("lazy")])])]),t._v(" "),a("dd",[t._v("If true, dataset instances are lazily loaded from disk, otherwise they are loaded and kept in memory.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("restore")])])]),t._v(" "),a("dd",[t._v("If enabled, tries to read previous training status from the "),a("code",[t._v("output")]),t._v(" folder and\ncontinues the training process")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("quiet")])])]),t._v(" "),a("dd",[t._v("If enabled, disables most logging messages keeping only warning and error messages.\nIn any case, all logging info will be stored into a file at ${output}/train.log")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("training_results")])]),t._v(" "),a("dd",[t._v("Training results including the generated model path and the related metrics")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"create-vocab"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#create-vocab"}},[t._v("#")]),t._v(" create_vocab "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("create_vocab")]),t._v(" ("),t._v("\n  self,\n  instance_datasets: Iterable[Union[allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset]],\n  config: Union["),a("a",{attrs:{title:"biome.text.configuration.VocabularyConfiguration",href:"configuration.html#biome.text.configuration.VocabularyConfiguration"}},[t._v("VocabularyConfiguration")]),t._v(", NoneType] = None,\n)  -> allennlp.data.vocabulary.Vocabulary\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Creates and updates the vocab of the pipeline.")]),t._v(" "),a("p",[t._v("NOTE: The trainer calls this method for you. You can use this method in case you want\nto create the vocab outside of the training process.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("instance_datasets")])])]),t._v(" "),a("dd",[t._v("A list of instance datasets from which to create the vocabulary.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("config")])])]),t._v(" "),a("dd",[t._v("Configurations for the vocab creation. Default: "),a("code",[t._v("VocabularyConfiguration()")]),t._v(".")])]),t._v(" "),a("h2",{attrs:{id:"examples"}},[t._v("Examples")]),t._v(" "),a("pre",[a("code",{staticClass:"language-python"},[t._v('>>> from biome.text import Pipeline, Dataset\n>>> pl = Pipeline.from_config(\n...     {"name": "example", "head":{"type": "TextClassification", "labels": ["pos", "neg"]}}\n... )\n>>> dataset = Dataset.from_dict({"text": ["Just an example"], "label": ["pos"]})\n>>> instance_dataset = dataset.to_instances(pl)\n>>> vocab = pl.create_vocab([instance_dataset])\n')])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"predict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predict"}},[t._v("#")]),t._v(" predict "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("predict")]),t._v(" ("),t._v("\n  self,\n  *args,\n  batch: Union[List[Dict[str, Any]], NoneType] = None,\n  add_tokens: bool = False,\n  add_attributions: bool = False,\n  attributions_kwargs: Union[Dict, NoneType] = None,\n  **kwargs,\n)  -> Union[Dict[str, numpy.ndarray], List[Union[Dict[str, numpy.ndarray], NoneType]]]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns a prediction given some input data based on the current state of the model")]),t._v(" "),a("p",[t._v("The accepted input is dynamically calculated and can be checked via the "),a("code",[t._v("self.inputs")]),t._v(" attribute\n("),a("code",[t._v("print("),a("a",{attrs:{title:"biome.text.pipeline.Pipeline.inputs",href:"#biome.text.pipeline.Pipeline.inputs"}},[t._v("Pipeline.inputs")]),t._v(")")]),t._v(")")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("em",[t._v("args/")]),t._v("*kwargs")]),t._v(" "),a("dt",[t._v("These are dynamically updated and correspond to the pipeline's "),a("code",[t._v("self.inputs")]),t._v(".")]),t._v(" "),a("dt",[t._v("If provided, the "),a("code",[t._v("batch")]),t._v(" parameter will be ignored.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("batch")])])]),t._v(" "),a("dd",[t._v("A list of dictionaries that represents a batch of inputs. The dictionary keys must comply with the\n"),a("code",[t._v("self.inputs")]),t._v(" attribute. Predicting batches should typically be faster than repeated calls with args/kwargs.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("add_tokens")])])]),t._v(" "),a("dd",[t._v("If true, adds a 'tokens' key in the prediction that contains the tokenized input.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("add_attributions")])])]),t._v(" "),a("dd",[t._v("If true, adds a 'attributions' key that contains attributions of the input to the prediction.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("attributions_kwargs")])])]),t._v(" "),a("dd",[t._v("This dict is directly passed on to the "),a("code",[t._v("TaskHead.compute_attributions()")]),t._v(".")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("predictions")])]),t._v(" "),a("dd",[t._v("A dictionary or a list of dictionaries containing the predictions and additional information.\nIf a prediction fails for a single input in the batch, its return value will be "),a("code",[t._v("None")]),t._v(".")])]),t._v(" "),a("h2",{attrs:{id:"raises"}},[t._v("Raises")]),t._v(" "),a("dl",[a("dt",[a("code",[a("a",{attrs:{title:"biome.text.pipeline.PredictionError",href:"#biome.text.pipeline.PredictionError"}},[t._v("PredictionError")])])]),t._v(" "),a("dd",[t._v("Failed to predict the single input or the whole batch")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"evaluate"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#evaluate"}},[t._v("#")]),t._v(" evaluate "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("evaluate")]),t._v(" ("),t._v("\n  self,\n  dataset: "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(",\n  batch_size: int = 16,\n  lazy: bool = False,\n  cuda_device: int = None,\n  predictions_output_file: Union[str, NoneType] = None,\n  metrics_output_file: Union[str, NoneType] = None,\n)  -> Dict[str, Any]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Evaluates the pipeline on a given dataset")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("dataset")])])]),t._v(" "),a("dd",[t._v("The dataset to use for the evaluation")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("batch_size")])])]),t._v(" "),a("dd",[t._v("Batch size used during the evaluation")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("lazy")])])]),t._v(" "),a("dd",[t._v("If true, instances from the dataset are lazily loaded from disk, otherwise they are loaded into memory.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("cuda_device")])])]),t._v(" "),a("dd",[t._v("If you want to use a specific CUDA device for the evaluation, specify it here. Pass on -1 for the CPU.\nBy default we will use a CUDA device if one is available.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("predictions_output_file")])])]),t._v(" "),a("dd",[t._v("Optional path to write the predictions to.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("metrics_output_file")])])]),t._v(" "),a("dd",[t._v("Optional path to write the final metrics to.")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("metrics")])]),t._v(" "),a("dd",[t._v("Metrics defined in the TaskHead")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"set-head"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#set-head"}},[t._v("#")]),t._v(" set_head "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("set_head")]),t._v(" ("),t._v("\n  self,\n  type: Type["),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")]),t._v("],\n  **kwargs,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Sets a new task head for the pipeline")]),t._v(" "),a("p",[t._v("Call this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("type")])]),t._v(" : "),a("code",[t._v("Type[TaskHead]")])]),t._v(" "),a("dd",[t._v("The "),a("code",[t._v("TaskHead")]),t._v(" class to be set for the pipeline (e.g., "),a("code",[t._v("TextClassification")])])]),t._v(" "),a("p",[t._v("**kwargs:\nThe "),a("code",[t._v("TaskHead")]),t._v(" specific arguments (e.g., the classification head needs a "),a("code",[t._v("pooler")]),t._v(" layer)")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"model-parameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#model-parameters"}},[t._v("#")]),t._v(" model_parameters "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("model_parameters")]),t._v("("),a("span",[t._v("self) -> Iterator[Tuple[str, torch.Tensor]]")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns an iterator over all model parameters, yielding the name and the parameter itself.")]),t._v(" "),a("h2",{attrs:{id:"examples"}},[t._v("Examples")]),t._v(" "),a("p",[t._v("You can use this to freeze certain parameters in the training:")]),t._v(" "),a("pre",[a("code",{staticClass:"language-python"},[t._v('>>> pipeline = Pipeline.from_config({\n...     "name": "model_parameters_example",\n...     "head": {"type": "TextClassification", "labels": ["a", "b"]},\n... })\n>>> for name, parameter in pipeline.model_parameters():\n...     if not name.endswith("bias"):\n...         parameter.requires_grad = False\n')])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"copy"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#copy"}},[t._v("#")]),t._v(" copy "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("copy")]),t._v("("),a("span",[t._v("self) -> "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Returns a copy of the pipeline")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"save"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#save"}},[t._v("#")]),t._v(" save "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("save")]),t._v(" ("),t._v("\n  self,\n  directory: Union[str, pathlib.Path],\n)  -> str\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Saves the pipeline in the given directory as "),a("code",[t._v("model.tar.gz")]),t._v(" file.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("directory")])])]),t._v(" "),a("dd",[t._v("Save the 'model.tar.gz' file to this directory.")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("file_path")])]),t._v(" "),a("dd",[t._v("Path to the 'model.tar.gz' file.")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"to-mlflow"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#to-mlflow"}},[t._v("#")]),t._v(" to_mlflow "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("to_mlflow")]),t._v(" ("),t._v("\n  self,\n  tracking_uri: Union[str, NoneType] = None,\n  experiment_id: Union[int, NoneType] = None,\n  run_name: str = 'log_biometext_model',\n  input_example: Union[Dict, NoneType] = None,\n  conda_env: Union[Dict, NoneType] = None,\n)  -> str\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Logs the pipeline as MLFlow Model to a MLFlow Tracking server")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("tracking_uri")])])]),t._v(" "),a("dd",[t._v("The URI of the MLFlow tracking server. MLFlow defaults to './mlruns'.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("experiment_id")])])]),t._v(" "),a("dd",[t._v("ID of the experiment under which to create the logging run. If this argument is unspecified,\nwill look for valid experiment in the following order: activated using "),a("code",[t._v("mlflow.set_experiment")]),t._v(",\n"),a("code",[t._v("MLFLOW_EXPERIMENT_NAME")]),t._v(" environment variable, "),a("code",[t._v("MLFLOW_EXPERIMENT_ID")]),t._v(" environment variable,\nor the default experiment as defined by the tracking server.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("run_name")])])]),t._v(" "),a("dd",[t._v("The name of the MLFlow run logging the model. Default: 'log_biometext_model'.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("input_example")])])]),t._v(" "),a("dd",[t._v("You can provide an input example in the form of a dictionary. For example, for a TextClassification head\nthis would be "),a("code",[t._v('{"text": "This is an input example"}')]),t._v(".")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("conda_env")])])]),t._v(" "),a("dd",[t._v("This conda environment is used when serving the model via "),a("code",[t._v("mlflow models serve")]),t._v('. Default:\nconda_env = {\n"name": "mlflow-dev",\n"channels": ["defaults", "conda-forge"],\n"dependencies": ["python=3.7.9", "pip>=20.3.0", {"pip": ["biome-text"]}],\n}')])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("model_uri")])]),t._v(" "),a("dd",[t._v("The URI of the logged MLFlow model. The model gets logged as an artifact to the corresponding run.")])]),t._v(" "),a("h2",{attrs:{id:"examples"}},[t._v("Examples")]),t._v(" "),a("p",[t._v("After logging the pipeline to MLFlow you can use the MLFlow model for inference:")]),t._v(" "),a("pre",[a("code",{staticClass:"language-python"},[t._v('>>> import mlflow, pandas, biome.text\n>>> pipeline = biome.text.Pipeline.from_config({\n...     "name": "to_mlflow_example",\n...     "head": {"type": "TextClassification", "labels": ["a", "b"]},\n... })\n>>> model_uri = pipeline.to_mlflow()\n>>> model = mlflow.pyfunc.load_model(model_uri)\n>>> prediction: pandas.DataFrame = model.predict(pandas.DataFrame([{"text": "Test this text"}]))\n')])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"create-vocabulary"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#create-vocabulary"}},[t._v("#")]),t._v(" create_vocabulary "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("create_vocabulary")]),t._v(" ("),t._v("\n  self,\n  config: "),a("a",{attrs:{title:"biome.text.configuration.VocabularyConfiguration",href:"configuration.html#biome.text.configuration.VocabularyConfiguration"}},[t._v("VocabularyConfiguration")]),t._v(",\n)  -> NoneType\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Creates the vocabulary for the pipeline from scratch")]),t._v(" "),a("p",[t._v("DEPRECATED: The vocabulary is now created automatically and this method will be removed in the future.\nYou can directly pass on a "),a("code",[t._v("VocabularyConfiguration")]),t._v(" to the "),a("code",[t._v("train")]),t._v(" method or use its default.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("config")])])]),t._v(" "),a("dd",[t._v("Specifies the sources of the vocabulary and how to extract it")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"predict-batch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predict-batch"}},[t._v("#")]),t._v(" predict_batch "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("predict_batch")]),t._v(" ("),t._v("\n  self,\n  *args,\n  **kwargs,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("DEPRECATED")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"explain"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#explain"}},[t._v("#")]),t._v(" explain "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("explain")]),t._v(" ("),t._v("\n  self,\n  *args,\n  **kwargs,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("DEPRECATED")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"explain-batch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#explain-batch"}},[t._v("#")]),t._v(" explain_batch "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("explain_batch")]),t._v(" ("),t._v("\n  self,\n  *args,\n  **kwargs,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("DEPRECATED")])])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"predictionerror"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predictionerror"}},[t._v("#")]),t._v(" PredictionError "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("PredictionError")]),t._v(" (...)"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Exception for a failed prediction of a single input or a whole batch")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[t._v("builtins.Exception")]),t._v(" "),a("li",[t._v("builtins.BaseException")])])])}),[],!1,null,null,null);e.default=n.exports}}]);