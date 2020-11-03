(window.webpackJsonp=window.webpackJsonp||[]).push([[35],{390:function(e,t,a){"use strict";a.r(t);var n=a(26),s=Object(n.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-hpo"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-hpo"}},[e._v("#")]),e._v(" biome.text.hpo "),a("Badge",{attrs:{text:"Module"}})],1),e._v(" "),a("div"),e._v(" "),a("p",[e._v("This module includes all components related to a HPO experiment execution.\nIt tries to allow for a simple integration with HPO libraries like Ray Tune.")]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"tune-hpo-train"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tune-hpo-train"}},[e._v("#")]),e._v(" tune_hpo_train "),a("Badge",{attrs:{text:"Function"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("tune_hpo_train")]),e._v(" ("),e._v("\n  config,\n  reporter,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("The main trainable method. This method defines common flow for hpo training.")]),e._v(" "),a("p",[e._v("See "),a("code",[a("a",{attrs:{title:"biome.text.hpo.HpoExperiment",href:"#biome.text.hpo.HpoExperiment"}},[e._v("HpoExperiment")])]),e._v(" for details about input parameters")])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"tunemetricslogger"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tunemetricslogger"}},[e._v("#")]),e._v(" TuneMetricsLogger "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("TuneMetricsLogger")]),e._v(" ()"),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("A trainer logger defined for sending validation metrics to ray tune system. Normally, those\nmetrics will be used by schedulers for trial experiments stop.")]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.loggers.BaseTrainLogger",href:"loggers.html#biome.text.loggers.BaseTrainLogger"}},[e._v("BaseTrainLogger")])]),e._v(" "),a("li",[e._v("allennlp.training.trainer.EpochCallback")]),e._v(" "),a("li",[e._v("allennlp.common.registrable.Registrable")]),e._v(" "),a("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members"}},[e._v("#")]),e._v(" Inherited members")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.loggers.BaseTrainLogger",href:"loggers.html#biome.text.loggers.BaseTrainLogger"}},[e._v("BaseTrainLogger")])])]),e._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.loggers.BaseTrainLogger.end_train",href:"loggers.html#biome.text.loggers.BaseTrainLogger.end_train"}},[e._v("end_train")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.loggers.BaseTrainLogger.init_train",href:"loggers.html#biome.text.loggers.BaseTrainLogger.init_train"}},[e._v("init_train")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.loggers.BaseTrainLogger.log_epoch_metrics",href:"loggers.html#biome.text.loggers.BaseTrainLogger.log_epoch_metrics"}},[e._v("log_epoch_metrics")])])])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"hpoparams"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hpoparams"}},[e._v("#")]),e._v(" HpoParams "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("HpoParams")]),e._v(" (pipeline: Dict[str, Any] = <factory>, trainer: Dict[str, Any] = <factory>)"),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("This class defines pipeline and trainer parameters selected for\nhyperparameter optimization sampling.")]),e._v(" "),a("h2",{attrs:{id:"attributes"}},[e._v("Attributes")]),e._v(" "),a("p",[e._v("pipeline:\nA selection of pipeline parameters used for tune sampling\ntrainer:\nA selection of trainer parameters used for tune sampling")]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"hpoexperiment"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hpoexperiment"}},[e._v("#")]),e._v(" HpoExperiment "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("HpoExperiment")]),e._v(" ("),e._v("\n    "),a("span",[e._v("name: str")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("pipeline: "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"pipeline.html#biome.text.pipeline.Pipeline"}},[e._v("Pipeline")])]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("train: str")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("validation: str")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("trainer: "),a("a",{attrs:{title:"biome.text.configuration.TrainerConfiguration",href:"configuration.html#biome.text.configuration.TrainerConfiguration"}},[e._v("TrainerConfiguration")]),e._v(" = <factory>")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("hpo_params: "),a("a",{attrs:{title:"biome.text.hpo.HpoParams",href:"#biome.text.hpo.HpoParams"}},[e._v("HpoParams")]),e._v(" = <factory>")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("shared_vocab: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("trainable_fn: Callable = <factory>")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("num_samples: int = 1")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("The hyper parameter optimization experiment data class")]),e._v(" "),a("h2",{attrs:{id:"attributes"}},[e._v("Attributes")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("name")])])]),e._v(" "),a("dd",[e._v("The experiment name used for experiment logging organization")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("pipeline")])])]),e._v(" "),a("dd",[a("code",[e._v("Pipeline")]),e._v(" used as base pipeline for hpo")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("train")])])]),e._v(" "),a("dd",[e._v("The train data source location")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("validation")])])]),e._v(" "),a("dd",[e._v("The validation data source location")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("trainer")])])]),e._v(" "),a("dd",[a("code",[e._v("TrainerConfiguration")]),e._v(" used as base trainer config for hpo")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("hpo_params")])])]),e._v(" "),a("dd",[a("code",[a("a",{attrs:{title:"biome.text.hpo.HpoParams",href:"#biome.text.hpo.HpoParams"}},[e._v("HpoParams")])]),e._v(" selected for hyperparameter sampling.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("shared_vocab")])])]),e._v(" "),a("dd",[e._v("If true, pipeline vocab will be used for all trials in this experiment.\nOtherwise, the vocab will be generated using input data sources in each trial.\nThis could be desired if some hpo defined param affects to vocab creation.\nDefaults: False")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("trainable_fn")])])]),e._v(" "),a("dd",[e._v("Function defining the hpo training flow. Normally the default function should\nbe enough for common use cases. Anyway, you can provide your own trainable function.\nIn this case, it's your responsibility to report tune metrics for a successful hpo\nDefaults: "),a("code",[a("a",{attrs:{title:"biome.text.hpo.tune_hpo_train",href:"#biome.text.hpo.tune_hpo_train"}},[e._v("tune_hpo_train()")])])]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("num_samples")])])]),e._v(" "),a("dd",[e._v("Number of times to sample from the hyperparameter space.\nIf "),a("code",[e._v("grid_search")]),e._v(" is provided as an argument in the hpo_params, the grid will be repeated "),a("code",[e._v("num_samples")]),e._v(" of times.\nDefault: 1")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"as-tune-experiment"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#as-tune-experiment"}},[e._v("#")]),e._v(" as_tune_experiment "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("as_tune_experiment")]),e._v("("),a("span",[e._v("self) -> ray.tune.experiment.Experiment")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd")]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"tuneexperiment"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tuneexperiment"}},[e._v("#")]),e._v(" TuneExperiment "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("TuneExperiment")]),e._v(" ("),e._v("\n    "),a("span",[e._v("pipeline_config: dict")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("trainer_config: dict")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("train_dataset: "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[e._v("Dataset")])]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("valid_dataset: "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[e._v("Dataset")])]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("vocab: Union[allennlp.data.vocabulary.Vocabulary, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("name: Union[str, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("trainable: Union[Callable, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("**kwargs")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("This class provides a trainable function and a config to conduct an HPO with "),a("code",[e._v("ray.tune.run")])]),e._v(" "),a("p",[e._v("Minimal usage:")]),e._v(" "),a("pre",[a("code",{staticClass:"language-python"},[e._v(">>> my_exp = TuneExperiment(pipeline_config, trainer_config, train_dataset, valid_dataset)\n>>> tune.run(my_exp)\n")])]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("pipeline_config")])])]),e._v(" "),a("dd",[e._v("The pipeline configuration with its hyperparemter search spaces:\n"),a("a",{attrs:{href:"https://docs.ray.io/en/master/tune/key-concepts.html#search-spaces"}},[e._v("https://docs.ray.io/en/master/tune/key-concepts.html#search-spaces")])]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("trainer_config")])])]),e._v(" "),a("dd",[e._v("The trainer configuration with its hyperparameter search spaces")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("train_dataset")])])]),e._v(" "),a("dd",[e._v("Training dataset")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("valid_dataset")])])]),e._v(" "),a("dd",[e._v("Validation dataset")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("vocab")])])]),e._v(" "),a("dd",[e._v("If you want to share the same vocabulary between the trials you can provide it here")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("name")])])]),e._v(" "),a("dd",[e._v("Used for the "),a("code",[e._v("tune.Experiment.name")]),e._v(", the project name in the WandB logger\nand for the experiment name in the MLFlow logger.\nBy default we construct following string: 'HPO on %date (%time)'")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("trainable")])])]),e._v(" "),a("dd",[e._v("A custom trainable function that takes as input the "),a("code",[a("a",{attrs:{title:"biome.text.hpo.TuneExperiment.config",href:"#biome.text.hpo.TuneExperiment.config"}},[e._v("TuneExperiment.config")])]),e._v(" dict.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("**kwargs")])])]),e._v(" "),a("dd",[e._v("The rest of the kwargs are passed on to "),a("code",[e._v("tune.Experiment.__init__")]),e._v(".\nThey must not contain the 'name', 'run' or the 'config' key,\nsince these are provided automatically by "),a("code",[a("a",{attrs:{title:"biome.text.hpo.TuneExperiment",href:"#biome.text.hpo.TuneExperiment"}},[e._v("TuneExperiment")])]),e._v(".")])]),e._v(" "),a("h2",{attrs:{id:"attributes"}},[e._v("Attributes")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("trainable")])])]),e._v(" "),a("dd",[e._v("The trainable function used by ray tune")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("config")])])]),e._v(" "),a("dd",[e._v("The config dict passed on to the trainable function")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[e._v("ray.tune.experiment.Experiment")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"instance-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[e._v("#")]),e._v(" Instance variables")]),e._v("\n")]),e._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.hpo.TuneExperiment.config"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("config")]),e._v(" : dict")])]),e._v(" "),a("dd",[a("p",[e._v("The config dictionary used by the "),a("code",[e._v("TuneExperiment.trainable")]),e._v(" function")])])])])}),[],!1,null,null,null);t.default=s.exports}}]);