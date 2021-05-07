(window.webpackJsonp=window.webpackJsonp||[]).push([[31],{437:function(e,t,a){"use strict";a.r(t);var n=a(26),s=Object(n.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-model"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-model"}},[e._v("#")]),e._v(" biome.text.model "),a("Badge",{attrs:{text:"Module"}})],1),e._v(" "),a("div"),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"pipelinemodel"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pipelinemodel"}},[e._v("#")]),e._v(" PipelineModel "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("PipelineModel")]),e._v(" (config: Dict, vocab: Union[allennlp.data.vocabulary.Vocabulary, NoneType] = None)"),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("This class represents pipeline model implementation for connect biome.text concepts with\nallennlp implementation details")]),e._v(" "),a("p",[e._v("This class manages the head + backbone encoder, keeping the allennlnlp Model lifecycle. This class\nshould be hidden to api users.")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("config")])])]),e._v(" "),a("dd",[e._v("Configuration of the pipeline")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("vocab")])])]),e._v(" "),a("dd",[e._v("The vocabulary of the pipeline. If None, an empty vocabulary will be created (default).")])]),e._v(" "),a("h2",{attrs:{id:"attributes"}},[e._v("Attributes")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("name")])]),e._v(" : "),a("code",[e._v("str")])]),e._v(" "),a("dd",[e._v("Name of the pipeline model")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("head")])]),e._v(" : "),a("code",[e._v("TaskHead")])]),e._v(" "),a("dd",[e._v("TaskHead of the pipeline model")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("vocab")])]),e._v(" : "),a("code",[e._v("Vocabulary")])]),e._v(" "),a("dd",[e._v("The vocabulary of the model, comes from allennlp.models.Model")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("file_path")])]),e._v(" : "),a("code",[e._v("Optional[str]")])]),e._v(" "),a("dd",[e._v("File path to a serialized version of this pipeline model")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("inputs")])]),e._v(" : "),a("code",[e._v("List[str]")])]),e._v(" "),a("dd",[e._v("The model inputs")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("output")])]),e._v(" : "),a("code",[e._v("List[str]")])]),e._v(" "),a("dd",[e._v("The model outputs (not prediction): Corresponding to the "),a("code",[e._v("TaskHead.featurize")]),e._v(" optional arguments.")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[e._v("allennlp.models.model.Model")]),e._v(" "),a("li",[e._v("pytorch_lightning.core.lightning.LightningModule")]),e._v(" "),a("li",[e._v("abc.ABC")]),e._v(" "),a("li",[e._v("pytorch_lightning.utilities.device_dtype_mixin.DeviceDtypeModuleMixin")]),e._v(" "),a("li",[e._v("pytorch_lightning.core.grads.GradInformation")]),e._v(" "),a("li",[e._v("pytorch_lightning.core.saving.ModelIO")]),e._v(" "),a("li",[e._v("pytorch_lightning.core.hooks.ModelHooks")]),e._v(" "),a("li",[e._v("pytorch_lightning.core.hooks.DataHooks")]),e._v(" "),a("li",[e._v("pytorch_lightning.core.hooks.CheckpointHooks")]),e._v(" "),a("li",[e._v("torch.nn.modules.module.Module")]),e._v(" "),a("li",[e._v("allennlp.common.registrable.Registrable")]),e._v(" "),a("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"from-params"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-params"}},[e._v("#")]),e._v(" from_params "),a("Badge",{attrs:{text:"Static method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("from_params")]),e._v(" ("),e._v("\n  params: allennlp.common.params.Params,\n  vocab: Union[allennlp.data.vocabulary.Vocabulary, NoneType] = None,\n  **extras,\n)  -> "),a("a",{attrs:{title:"biome.text.model.PipelineModel",href:"#biome.text.model.PipelineModel"}},[e._v("PipelineModel")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Load the model implementation from params. We build manually each component from config sections.")]),e._v(" "),a("p",[e._v("The param keys matches exactly with keys in yaml configuration files")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("params")])])]),e._v(" "),a("dd",[e._v("The config key in these params is used to build the model components")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("vocab")])])]),e._v(" "),a("dd",[e._v("The vocabulary for the model")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("**extras")])])]),e._v(" "),a("dd",[e._v("Necessary for AllenNLP from_params machinery")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("pipeline_model")])]),e._v(" "),a("dd",[e._v(" ")])])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"instance-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[e._v("#")]),e._v(" Instance variables")]),e._v("\n")]),e._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.model.PipelineModel.head"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("head")]),e._v(" : "),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[e._v("TaskHead")])])]),e._v(" "),a("dd",[a("p",[e._v("Get the model head")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.model.PipelineModel.inputs"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("inputs")]),e._v(" : List[str]")])]),e._v(" "),a("dd",[a("p",[e._v("The model inputs. Corresponding to head.featurize required argument names")])]),e._v(" "),a("dt",{attrs:{id:"biome.text.model.PipelineModel.output"}},[a("code",{staticClass:"name"},[e._v("var "),a("span",{staticClass:"ident"},[e._v("output")]),e._v(" : List[str]")])]),e._v(" "),a("dd",[a("p",[e._v("The model outputs (not prediction): Corresponding to head.featurize optional argument names.")])])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"set-head"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#set-head"}},[e._v("#")]),e._v(" set_head "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("set_head")]),e._v(" ("),e._v("\n  self,\n  head: "),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[e._v("TaskHead")]),e._v(",\n)  -> NoneType\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Set a head and update related model attributes")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"forward"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#forward"}},[e._v("#")]),e._v(" forward "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("forward")]),e._v(" ("),e._v("\n  self,\n  *args,\n  **kwargs,\n)  -> Dict[str, torch.Tensor]\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("The main forward method just wraps the head forward method")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"get-metrics"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-metrics"}},[e._v("#")]),e._v(" get_metrics "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("get_metrics")]),e._v(" ("),e._v("\n  self,\n  reset: bool = False,\n)  -> Dict[str, float]\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Fetch metrics defined in head layer")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"text-to-instance"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#text-to-instance"}},[e._v("#")]),e._v(" text_to_instance "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("text_to_instance")]),e._v(" ("),e._v("\n  self,\n  **inputs,\n)  -> Union[allennlp.data.instance.Instance, NoneType]\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Applies the head featurize method")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"extend-vocabulary"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#extend-vocabulary"}},[e._v("#")]),e._v(" extend_vocabulary "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("extend_vocabulary")]),e._v(" ("),e._v("\n  self,\n  vocab: allennlp.data.vocabulary.Vocabulary,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Extend the model's vocabulary with "),a("code",[e._v("vocab")])]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("vocab")])])]),e._v(" "),a("dd",[e._v("The model's vocabulary will be extended with this one.")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"extend-embedder-vocab"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#extend-embedder-vocab"}},[e._v("#")]),e._v(" extend_embedder_vocab "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("extend_embedder_vocab")]),e._v(" ("),e._v("\n  self,\n  embedding_sources_mapping: Dict[str, str] = None,\n)  -> NoneType\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Iterates through all embedding modules in the model and assures it can embed\nwith the extended vocab. This is required in fine-tuning or transfer learning\nscenarios where model was trained with original vocabulary but during\nfine-tuning/transfer-learning, it will have it work with extended vocabulary\n(original + new-data vocabulary).")]),e._v(" "),a("h1",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("p",[e._v("embedding_sources_mapping : "),a("code",[e._v("Dict[str, str]")]),e._v(", optional (default = "),a("code",[e._v("None")]),e._v(')\nMapping from model_path to pretrained-file path of the embedding\nmodules. If pretrained-file used at time of embedding initialization\nisn\'t available now, user should pass this mapping. Model path is\npath traversing the model attributes upto this embedding module.\nEg. "_text_field_embedder.token_embedder_tokens".')])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-logger"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-logger"}},[e._v("#")]),e._v(" init_prediction_logger "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("init_prediction_logger")]),e._v(" ("),e._v("\n  self,\n  output_dir: str,\n  max_bytes: int = 20000000,\n  backup_count: int = 20,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Initialize the prediction logger.")]),e._v(" "),a("p",[e._v("If initialized we will log all predictions to a file called "),a("em",[e._v("predictions.json")]),e._v(" in the "),a("code",[e._v("output_folder")]),e._v(".")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("output_dir")])])]),e._v(" "),a("dd",[e._v("Path to the folder in which we create the "),a("em",[e._v("predictions.json")]),e._v(" file.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("max_bytes")])])]),e._v(" "),a("dd",[e._v("Passed on to logging.handlers.RotatingFileHandler")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("backup_count")])])]),e._v(" "),a("dd",[e._v("Passed on to logging.handlers.RotatingFileHandler")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"init-prediction-cache"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#init-prediction-cache"}},[e._v("#")]),e._v(" init_prediction_cache "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("init_prediction_cache")]),e._v(" ("),e._v("\n  self,\n  max_size: int,\n)  -> NoneType\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Initialize a prediction cache using the functools.lru_cache decorator")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("max_size")])])]),e._v(" "),a("dd",[e._v("Save up to max_size most recent items.")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"predict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predict"}},[e._v("#")]),e._v(" predict "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("predict")]),e._v(" ("),e._v("\n  self,\n  batch: List[Dict[str, Union[str, List[str], Dict[str, str]]]],\n  prediction_config: "),a("a",{attrs:{title:"biome.text.configuration.PredictionConfiguration",href:"configuration.html#biome.text.configuration.PredictionConfiguration"}},[e._v("PredictionConfiguration")]),e._v(",\n)  -> List[Union["),a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.TaskPrediction",href:"modules/heads/task_prediction.html#biome.text.modules.heads.task_prediction.TaskPrediction"}},[e._v("TaskPrediction")]),e._v(", NoneType]]\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Returns predictions given some input data based on the current state of the model")]),e._v(" "),a("p",[e._v("The keys of the input dicts in the batch must coincide with the "),a("code",[e._v("self.inputs")]),e._v(" attribute.\nTODO: Comply with LightningModule API + Trainer API (means move instance creation logic to Pipeline)")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("batch")])])]),e._v(" "),a("dd",[e._v("A list of dictionaries that represents a batch of inputs.\nThe dictionary keys must comply with the "),a("code",[e._v("self.inputs")]),e._v(" attribute.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("prediction_config")])])]),e._v(" "),a("dd",[e._v("Contains configurations for the prediction")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"on-fit-start"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#on-fit-start"}},[e._v("#")]),e._v(" on_fit_start "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("on_fit_start")]),e._v("("),a("span",[e._v("self) -> NoneType")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Called at the very beginning of fit.\nIf on DDP it is called on every process")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"training-step"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#training-step"}},[e._v("#")]),e._v(" training_step "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("training_step")]),e._v(" ("),e._v("\n  self,\n  batch,\n  batch_idx,\n)  -> Dict\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Here you compute and return the training loss and some additional metrics for e.g.\nthe progress bar or logger.")]),e._v(" "),a("p",[e._v("Args:\nbatch (:class:"),a("code",[e._v("~torch.Tensor")]),e._v(" | (:class:"),a("code",[e._v("~torch.Tensor")]),e._v(", …) | [:class:"),a("code",[e._v("~torch.Tensor")]),e._v(", …]):\nThe output of your :class:"),a("code",[e._v("~torch.utils.data.DataLoader")]),e._v(". A tensor, tuple or list.\nbatch_idx (int): Integer displaying index of this batch\noptimizer_idx (int): When using multiple optimizers, this argument will also be present.\nhiddens(:class:"),a("code",[e._v("~torch.Tensor")]),e._v("): Passed in if\n:paramref:"),a("code",[e._v("~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps")]),e._v(" > 0.")]),e._v(" "),a("p",[e._v("Return:\nAny of.")]),e._v(" "),a("pre",[a("code",[e._v("- :class:`~torch.Tensor` - The loss tensor\n- <code>dict</code> - A dictionary. Can include any keys, but must include the key ``'loss'``\n- <code>None</code> - Training will skip to the next batch\n")])]),e._v(" "),a("p",[e._v("Note:\nReturning "),a("code",[e._v("None")]),e._v(" is currently not supported for multi-GPU or TPU, or with 16-bit precision enabled.")]),e._v(" "),a("p",[e._v("In this step you'd normally do the forward pass and calculate the loss for a batch.\nYou can also do fancier things like multiple forward passes or something model specific.")]),e._v(" "),a("p",[e._v("Example::")]),e._v(" "),a("pre",[a("code",[e._v("def training_step(self, batch, batch_idx):\n    x, y, z = batch\n    out = self.encoder(x)\n    loss = self.loss(out, x)\n    return loss\n")])]),e._v(" "),a("p",[e._v("If you define multiple optimizers, this step will be called with an additional\n"),a("code",[e._v("optimizer_idx")]),e._v(" parameter.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# Multiple optimizers (e.g.: GANs)\ndef training_step(self, batch, batch_idx, optimizer_idx):\n    if optimizer_idx == 0:\n        # do training_step with encoder\n    if optimizer_idx == 1:\n        # do training_step with decoder\n")])]),e._v(" "),a("p",[e._v("If you add truncated back propagation through time you will also get an additional\nargument with the hidden states of the previous step.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# Truncated back-propagation through time\ndef training_step(self, batch, batch_idx, hiddens):\n    # hiddens are the hidden states from the previous truncated backprop step\n    ...\n    out, hiddens = self.lstm(data, hiddens)\n    ...\n    return {'loss': loss, 'hiddens': hiddens}\n")])]),e._v(" "),a("p",[e._v("Note:\nThe loss value shown in the progress bar is smoothed (averaged) over the last values,\nso it differs from the actual loss returned in train/validation step.")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"training-epoch-end"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#training-epoch-end"}},[e._v("#")]),e._v(" training_epoch_end "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("training_epoch_end")]),e._v(" ("),e._v("\n  self,\n  outputs: List[Any],\n)  -> NoneType\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Called at the end of the training epoch with the outputs of all training steps.\nUse this in case you need to do something with all the outputs for every training_step.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# the pseudocode for these calls\ntrain_outs = []\nfor train_batch in train_data:\n    out = training_step(train_batch)\n    train_outs.append(out)\ntraining_epoch_end(train_outs)\n")])]),e._v(" "),a("p",[e._v("Args:\noutputs: List of outputs you defined in :meth:"),a("code",[e._v("training_step")]),e._v(", or if there are\nmultiple dataloaders, a list containing a list of outputs for each dataloader.")]),e._v(" "),a("p",[e._v("Return:\nNone")]),e._v(" "),a("p",[e._v("Note:\nIf this method is not overridden, this won't be called.")]),e._v(" "),a("p",[e._v("Example::")]),e._v(" "),a("pre",[a("code",[e._v("def training_epoch_end(self, training_step_outputs):\n    # do something with all training_step outputs\n    return result\n")])]),e._v(" "),a("p",[e._v("With multiple dataloaders, "),a("code",[e._v("outputs")]),e._v(" will be a list of lists. The outer list contains\none entry per dataloader, while the inner list contains the individual outputs of\neach training step for that dataloader.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("def training_epoch_end(self, training_step_outputs):\n    for out in training_step_outputs:\n        # do something here\n")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"validation-step"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#validation-step"}},[e._v("#")]),e._v(" validation_step "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("validation_step")]),e._v(" ("),e._v("\n  self,\n  batch,\n  batch_idx,\n)  -> Dict\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Operates on a single batch of data from the validation set.\nIn this step you'd might generate examples or calculate anything of interest like accuracy.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# the pseudocode for these calls\nval_outs = []\nfor val_batch in val_data:\n    out = validation_step(val_batch)\n    val_outs.append(out)\nvalidation_epoch_end(val_outs)\n")])]),e._v(" "),a("p",[e._v("Args:\nbatch (:class:"),a("code",[e._v("~torch.Tensor")]),e._v(" | (:class:"),a("code",[e._v("~torch.Tensor")]),e._v(", …) | [:class:"),a("code",[e._v("~torch.Tensor")]),e._v(", …]):\nThe output of your :class:"),a("code",[e._v("~torch.utils.data.DataLoader")]),e._v(". A tensor, tuple or list.\nbatch_idx (int): The index of this batch\ndataloader_idx (int): The index of the dataloader that produced this batch\n(only if multiple val dataloaders used)")]),e._v(" "),a("p",[e._v("Return:\nAny of.")]),e._v(" "),a("pre",[a("code",[e._v("- Any object or value\n- <code>None</code> - Validation will skip to the next batch\n")])]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# pseudocode of order\nval_outs = []\nfor val_batch in val_data:\n    out = validation_step(val_batch)\n    if defined('validation_step_end'):\n        out = validation_step_end(out)\n    val_outs.append(out)\nval_outs = validation_epoch_end(val_outs)\n")])]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# if you have one val dataloader:\ndef validation_step(self, batch, batch_idx)\n"),a("h1",{attrs:{id:"if-you-have-multiple-val-dataloaders"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#if-you-have-multiple-val-dataloaders"}},[e._v("#")]),e._v(" if you have multiple val dataloaders:")]),e._v("\n"),a("p",[e._v("def validation_step(self, batch, batch_idx, dataloader_idx)\n")])])]),e._v(" "),a("p",[e._v("Examples::")]),e._v(" "),a("pre",[a("code",[e._v("# CASE 1: A single validation dataset\ndef validation_step(self, batch, batch_idx):\nx, y = batch"),a("p"),e._v("\n"),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v("# implement your own\nout = self(x)\nloss = self.loss(out, y)\n\n# log 6 example images\n# or generated text... or whatever\nsample_imgs = x[:6]\ngrid = torchvision.utils.make_grid(sample_imgs)\nself.logger.experiment.add_image('example_images', grid, 0)\n\n# calculate acc\nlabels_hat = torch.argmax(out, dim=1)\nval_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)\n\n# log the outputs!\nself.log_dict({'val_loss': loss, 'val_acc': val_acc})\n")])])]),a("p")])]),e._v(" "),a("p",[e._v("If you pass in multiple val dataloaders, :meth:"),a("code",[e._v("validation_step")]),e._v(" will have an additional argument.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# CASE 2: multiple validation dataloaders\ndef validation_step(self, batch, batch_idx, dataloader_idx):\n# dataloader_idx tells you which dataset this is.\n")])]),e._v(" "),a("p",[e._v("Note:\nIf you don't need to validate you don't need to implement this method.")]),e._v(" "),a("p",[e._v("Note:\nWhen the :meth:"),a("code",[e._v("validation_step")]),e._v(" is called, the model has been put in eval mode\nand PyTorch gradients have been disabled. At the end of validation,\nthe model goes back to training mode and gradients are enabled.")])]),e._v(" "),a("pre",{staticClass:"title"},[a("p"),e._v("\n"),a("h3",{attrs:{id:"validation-epoch-end"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#validation-epoch-end"}},[e._v("#")]),e._v(" validation_epoch_end "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("validation_epoch_end")]),e._v(" ("),e._v("\n  self,\n  outputs: List[Any],\n)  -> NoneType\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Called at the end of the validation epoch with the outputs of all validation steps.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("pre",[a("code",[e._v("# the pseudocode for these calls\nval_outs = []\nfor val_batch in val_data:\n    out = validation_step(val_batch)\n    val_outs.append(out)\nvalidation_epoch_end(val_outs)\n")])]),e._v(" "),a("p",[e._v("Args:\noutputs: List of outputs you defined in :meth:"),a("code",[e._v("validation_step")]),e._v(", or if there\nare multiple dataloaders, a list containing a list of outputs for each dataloader.")]),e._v(" "),a("p",[e._v("Return:\nNone")]),e._v(" "),a("p",[e._v("Note:\nIf you didn't define a :meth:"),a("code",[e._v("validation_step")]),e._v(", this won't be called.")]),e._v(" "),a("p",[e._v("Examples:\nWith a single dataloader:")]),e._v(" "),a("pre",[a("code",[e._v(".. code-block:: python\n"),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v("def validation_epoch_end(self, val_step_outputs):\n    for out in val_step_outputs:\n        # do something\n")])])]),a("p",[e._v("With multiple dataloaders, <code>outputs</code> will be a list of lists. The outer list contains\none entry per dataloader, while the inner list contains the individual outputs of\neach validation step for that dataloader.")]),e._v(" "),a("p",[e._v(".. code-block:: python")]),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v("def validation_epoch_end(self, outputs):\n    for dataloader_output_result in outputs:\n        dataloader_outs = dataloader_output_result.dataloader_i_outputs\n\n    self.log('final_metric', final_value)\n")])])]),a("p")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("p"),e._v("\n"),a("h3",{attrs:{id:"configure-optimizers"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#configure-optimizers"}},[e._v("#")]),e._v(" configure_optimizers "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("configure_optimizers")]),e._v("("),a("span",[e._v("self)")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Choose what optimizers and learning-rate schedulers to use in your optimization.\nNormally you'd need one. But in the case of GANs or similar you might have multiple.")]),e._v(" "),a("p",[e._v("Return:\nAny of these 6 options.")]),e._v(" "),a("pre",[a("code",[e._v('- **Single optimizer**.\n- **List or Tuple** of optimizers.\n- **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers (or\n  multiple lr_dict).\n- **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``\n  key whose value is a single LR scheduler or lr_dict.\n- **Tuple of dictionaries** as described above, with an optional ``"frequency"`` key.\n- **None** - Fit will run without any optimizer.\n')])]),e._v(" "),a("p",[e._v("Note:\nThe lr_dict is a dictionary which contains the scheduler and its associated configuration.\nThe default configuration is shown below.")]),e._v(" "),a("pre",[a("code",[e._v(".. code-block:: python\n"),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v("lr_dict = {\n    'scheduler': lr_scheduler, # The LR scheduler instance (required)\n    # The unit of the scheduler's step size, could also be 'step'\n    'interval': 'epoch',\n    'frequency': 1, # The frequency of the scheduler\n    'monitor': 'val_loss', # Metric for &lt;code&gt;ReduceLROnPlateau&lt;/code&gt; to monitor\n    'strict': True, # Whether to crash the training if &lt;code&gt;monitor&lt;/code&gt; is not found\n    'name': None, # Custom name for &lt;code&gt;LearningRateMonitor&lt;/code&gt; to use\n}\n")])])]),a("p",[e._v("Only the "),a("code",[e._v('"scheduler"')]),e._v(" key is required, the rest will be set to the defaults above.\n")])])]),e._v(" "),a("p",[e._v("Note:\nThe "),a("code",[e._v("frequency")]),e._v(" value specified in a dict along with the "),a("code",[e._v("optimizer")]),e._v(" key is an int corresponding\nto the number of sequential batches optimized with the specific optimizer.\nIt should be given to none or to all of the optimizers.\nThere is a difference between passing multiple optimizers in a list,\nand passing multiple optimizers in dictionaries with a frequency of 1:\nIn the former case, all optimizers will operate on the given batch in each optimization step.\nIn the latter, only one optimizer will operate on the given batch at every step.\nThis is different from the "),a("code",[e._v("frequency")]),e._v(" value specified in the lr_dict mentioned below.")]),e._v(" "),a("pre",[a("code",[e._v(".. code-block:: python"),a("p"),e._v("\n"),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v("def configure_optimizers(self):\n    optimizer_one = torch.optim.SGD(self.model.parameters(), lr=0.01)\n    optimizer_two = torch.optim.SGD(self.model.parameters(), lr=0.01)\n    return [\n        {'optimizer': optimizer_one, 'frequency': 5},\n        {'optimizer': optimizer_two, 'frequency': 10},\n    ]\n")])])]),a("p",[e._v("In this example, the first optimizer will be used for the first 5 steps,\nthe second optimizer for the next 10 steps and that cycle will continue.\nIf an LR scheduler is specified for an optimizer using the <code>lr_scheduler</code> key in the above dict,\nthe scheduler will only be updated when its optimizer is being used.\n")])])]),e._v(" "),a("p",[e._v("Examples::")]),e._v(" "),a("pre",[a("code",[e._v("# most cases\ndef configure_optimizers(self):\nreturn Adam(self.parameters(), lr=1e-3)"),a("p"),e._v("\n"),a("h1",{attrs:{id:"multiple-optimizer-case-e-g-gan"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#multiple-optimizer-case-e-g-gan"}},[e._v("#")]),e._v(" multiple optimizer case (e.g.: GAN)")]),e._v("\n"),a("p",[e._v("def configure_optimizers(self):\ngen_opt = Adam(self.model_gen.parameters(), lr=0.01)\ndis_opt = Adam(self.model_dis.parameters(), lr=0.02)\nreturn gen_opt, dis_opt")]),e._v("\n"),a("h1",{attrs:{id:"example-with-learning-rate-schedulers"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#example-with-learning-rate-schedulers"}},[e._v("#")]),e._v(" example with learning rate schedulers")]),e._v("\n"),a("p",[e._v("def configure_optimizers(self):\ngen_opt = Adam(self.model_gen.parameters(), lr=0.01)\ndis_opt = Adam(self.model_dis.parameters(), lr=0.02)\ndis_sch = CosineAnnealing(dis_opt, T_max=10)\nreturn [gen_opt, dis_opt], [dis_sch]")]),e._v("\n"),a("h1",{attrs:{id:"example-with-step-based-learning-rate-schedulers"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#example-with-step-based-learning-rate-schedulers"}},[e._v("#")]),e._v(" example with step-based learning rate schedulers")]),e._v("\n"),a("p",[e._v("def configure_optimizers(self):\ngen_opt = Adam(self.model_gen.parameters(), lr=0.01)\ndis_opt = Adam(self.model_dis.parameters(), lr=0.02)\ngen_sch = {'scheduler': ExponentialLR(gen_opt, 0.99),\n'interval': 'step'}  # called after each training step\ndis_sch = CosineAnnealing(dis_opt, T_max=10) # called every epoch\nreturn [gen_opt, dis_opt], [gen_sch, dis_sch]")]),e._v("\n"),a("h1",{attrs:{id:"example-with-optimizer-frequencies"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#example-with-optimizer-frequencies"}},[e._v("#")]),e._v(" example with optimizer frequencies")]),e._v("\n"),a("h1",{attrs:{id:"see-training-procedure-in-code-improved-training-of-wasserstein-gans-code-algorithm-1"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#see-training-procedure-in-code-improved-training-of-wasserstein-gans-code-algorithm-1"}},[e._v("#")]),e._v(" see training procedure in <code>Improved Training of Wasserstein GANs</code>, Algorithm 1")]),e._v("\n"),a("h1",{attrs:{id:"https-arxiv-org-abs-1704-00028"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#https-arxiv-org-abs-1704-00028"}},[e._v("#")]),e._v(" <https://arxiv.org/abs/1704.00028>")]),e._v("\n"),a("p",[e._v("def configure_optimizers(self):\ngen_opt = Adam(self.model_gen.parameters(), lr=0.01)\ndis_opt = Adam(self.model_dis.parameters(), lr=0.02)\nn_critic = 5\nreturn (\n{'optimizer': dis_opt, 'frequency': n_critic},\n{'optimizer': gen_opt, 'frequency': 1}\n)\n")])])]),e._v(" "),a("p",[e._v("Note:\nSome things to know:")]),e._v(" "),a("pre",[a("code",[e._v("- Lightning calls <code>.backward()</code> and <code>.step()</code> on each optimizer and learning rate scheduler as needed."),a("p"),e._v("\n"),a("ul",[e._v("\n"),a("li",[e._v("If you use 16-bit precision ("),a("code",[e._v("precision=16")]),e._v("), Lightning will automatically handle the optimizers.")]),e._v("\n"),a("li",[e._v("If you use multiple optimizers, :meth:<code>training_step</code> will have an additional <code>optimizer_idx</code> parameter.")]),e._v("\n"),a("li",[e._v("If you use :class:<code>torch.optim.LBFGS</code>, Lightning handles the closure function automatically for you.")]),e._v("\n"),a("li",[e._v("If you use multiple optimizers, gradients will be calculated only for the parameters of current optimizer\nat each training step.")]),e._v("\n"),a("li",[e._v("If you need to control how often those optimizers step or override the default <code>.step()</code> schedule,\noverride the :meth:<code>optimizer_step</code> hook.\n")])])])])])])])}),[],!1,null,null,null);t.default=s.exports}}]);