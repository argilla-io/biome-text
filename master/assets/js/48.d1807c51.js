(window.webpackJsonp=window.webpackJsonp||[]).push([[48],{453:function(t,e,a){"use strict";a.r(e);var i=a(26),s=Object(i.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-modules-heads-classification-text-classification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-classification-text-classification"}},[t._v("#")]),t._v(" biome.text.modules.heads.classification.text_classification "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"textclassification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#textclassification"}},[t._v("#")]),t._v(" TextClassification "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TextClassification")]),t._v(" ("),t._v("\n    "),a("span",[t._v("backbone: "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"../../../backbone.html#biome.text.backbone.ModelBackbone"}},[t._v("ModelBackbone")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("labels: List[str]")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("pooler: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration"}},[t._v("Seq2VecEncoderConfiguration")]),t._v(", NoneType] = None")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("feedforward: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration"}},[t._v("FeedForwardConfiguration")]),t._v(", NoneType] = None")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("multilabel: bool = False")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("label_weights: Union[List[float], Dict[str, float], NoneType] = None")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Task head for text classification")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("backbone")])])]),t._v(" "),a("dd",[t._v("The backbone of your model. Must not be provided when initiating with "),a("code",[t._v("Pipeline.from_config")]),t._v(".")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("labels")])])]),t._v(" "),a("dd",[t._v("A list of labels for your classification task.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("pooler")])])]),t._v(" "),a("dd",[t._v("The pooler of the output sequence from the backbone model. Default: "),a("code",[t._v("BagOfEmbeddingsEncoder")]),t._v(".")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("feedforward")])])]),t._v(" "),a("dd",[t._v("An optional feedforward layer applied to the output of the pooler. Default: None.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("multilabel")])])]),t._v(" "),a("dd",[t._v("Is this a multi label classification task? Default: False")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("label_weights")])])]),t._v(" "),a("dd",[t._v("A list of weights for each label. The weights must be in the same order as the "),a("code",[t._v("labels")]),t._v(".\nYou can also provide a dictionary that maps the label to its weight. Default: None.")])]),t._v(" "),a("p",[t._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead"}},[t._v("ClassificationHead")])]),t._v(" "),a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")])]),t._v(" "),a("li",[t._v("torch.nn.modules.module.Module")]),t._v(" "),a("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members"}},[t._v("#")]),t._v(" Inherited members")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead"}},[t._v("ClassificationHead")])])]),t._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.extend_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels"}},[t._v("extend_labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.featurize",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.featurize"}},[t._v("featurize")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.forward",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.forward"}},[t._v("forward")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics"}},[t._v("get_metrics")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.inputs",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.inputs"}},[t._v("inputs")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.labels"}},[t._v("labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.make_task_prediction",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.make_task_prediction"}},[t._v("make_task_prediction")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.num_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels"}},[t._v("num_labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.on_vocab_update",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update"}},[t._v("on_vocab_update")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.register",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.register"}},[t._v("register")])])])])])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"textclassificationconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#textclassificationconfiguration"}},[t._v("#")]),t._v(" TextClassificationConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TextClassificationConfiguration")]),t._v(" (*args, **kwds)"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Configuration for classification head components")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[t._v("ComponentConfiguration")])]),t._v(" "),a("li",[t._v("typing.Generic")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members-2"}},[t._v("#")]),t._v(" Inherited members")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[t._v("ComponentConfiguration")])])]),t._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.compile",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.compile"}},[t._v("compile")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.config",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.config"}},[t._v("config")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.from_params",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.from_params"}},[t._v("from_params")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.input_dim",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.input_dim"}},[t._v("input_dim")])])])])])])])}),[],!1,null,null,null);e.default=s.exports}}]);