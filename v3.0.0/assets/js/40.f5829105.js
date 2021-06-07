(window.webpackJsonp=window.webpackJsonp||[]).push([[40],{446:function(e,t,a){"use strict";a.r(t);var i=a(26),o=Object(i.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-modules-heads-classification-record-classification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-classification-record-classification"}},[e._v("#")]),e._v(" biome.text.modules.heads.classification.record_classification "),a("Badge",{attrs:{text:"Module"}})],1),e._v(" "),a("div"),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"recordclassification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#recordclassification"}},[e._v("#")]),e._v(" RecordClassification "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("RecordClassification")]),e._v(" ("),e._v("\n    "),a("span",[e._v("backbone: "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"../../../backbone.html#biome.text.backbone.ModelBackbone"}},[e._v("ModelBackbone")])]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("labels: List[str]")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("record_keys: List[str]")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("token_pooler: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration"}},[e._v("Seq2VecEncoderConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("fields_encoder: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration"}},[e._v("Seq2SeqEncoderConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("fields_pooler: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration"}},[e._v("Seq2VecEncoderConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("feedforward: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration"}},[e._v("FeedForwardConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("multilabel: Union[bool, NoneType] = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("label_weights: Union[List[float], Dict[str, float], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Task head for data record\nclassification.\nAccepts a variable data inputs and apply featuring over defined record keys.")]),e._v(" "),a("p",[e._v("This head applies a doc2vec architecture from a structured record data input")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("backbone")])])]),e._v(" "),a("dd",[e._v("The backbone of your model. Must not be provided when initiating with "),a("code",[e._v("Pipeline.from_config")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("labels")])])]),e._v(" "),a("dd",[e._v("A list of labels for your classification task.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("token_pooler")])])]),e._v(" "),a("dd",[e._v("The pooler at token level to provide one vector per record field. Default: "),a("code",[e._v("BagOfEmbeddingsEncoder")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("fields_encoder")])])]),e._v(" "),a("dd",[e._v("An optional sequence to sequence encoder that contextualizes the record field representations. Default: None.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("fields_pooler")])])]),e._v(" "),a("dd",[e._v("The pooler at sentence level to provide a vector for the whole record. Default: "),a("code",[e._v("BagOfEmbeddingsEncoder")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("feedforward")])])]),e._v(" "),a("dd",[e._v("An optional feedforward layer applied to the output of the fields pooler. Default: None.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("multilabel")])])]),e._v(" "),a("dd",[e._v("Is this a multi label classification task? Default: False")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("label_weights")])])]),e._v(" "),a("dd",[e._v("A list of weights for each label. The weights must be in the same order as the "),a("code",[e._v("labels")]),e._v(".\nYou can also provide a dictionary that maps the label to its weight. Default: None.")])]),e._v(" "),a("p",[e._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification",href:"doc_classification.html#biome.text.modules.heads.classification.doc_classification.DocumentClassification"}},[e._v("DocumentClassification")])]),e._v(" "),a("li",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead"}},[e._v("ClassificationHead")])]),e._v(" "),a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[e._v("TaskHead")])]),e._v(" "),a("li",[e._v("torch.nn.modules.module.Module")]),e._v(" "),a("li",[e._v("allennlp.common.registrable.Registrable")]),e._v(" "),a("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inputs"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inputs"}},[e._v("#")]),e._v(" inputs "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("inputs")]),e._v("("),a("span",[e._v("self) -> Union[List[str], NoneType]")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("The inputs names are determined by configured record keys")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members"}},[e._v("#")]),e._v(" Inherited members")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification",href:"doc_classification.html#biome.text.modules.heads.classification.doc_classification.DocumentClassification"}},[e._v("DocumentClassification")])])]),e._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.extend_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels"}},[e._v("extend_labels")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.featurize",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.featurize"}},[e._v("featurize")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.forward",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.forward"}},[e._v("forward")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.get_metrics",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics"}},[e._v("get_metrics")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.labels"}},[e._v("labels")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.make_task_prediction",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.make_task_prediction"}},[e._v("make_task_prediction")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.num_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels"}},[e._v("num_labels")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.on_vocab_update",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update"}},[e._v("on_vocab_update")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.doc_classification.DocumentClassification.register",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.register"}},[e._v("register")])])])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"recordclassificationconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#recordclassificationconfiguration"}},[e._v("#")]),e._v(" RecordClassificationConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("RecordClassificationConfiguration")]),e._v(" (*args, **kwds)"),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Lazy initialization for document classification head components")]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[e._v("ComponentConfiguration")])]),e._v(" "),a("li",[e._v("typing.Generic")]),e._v(" "),a("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members-2"}},[e._v("#")]),e._v(" Inherited members")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[e._v("ComponentConfiguration")])])]),e._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.compile",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.compile"}},[e._v("compile")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.config",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.config"}},[e._v("config")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.from_params",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.from_params"}},[e._v("from_params")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.input_dim",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.input_dim"}},[e._v("input_dim")])])])])])])])}),[],!1,null,null,null);t.default=o.exports}}]);