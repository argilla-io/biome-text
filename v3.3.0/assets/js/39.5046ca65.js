(window.webpackJsonp=window.webpackJsonp||[]).push([[39],{456:function(e,t,a){"use strict";a.r(t);var i=a(28),o=Object(i.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-modules-heads-classification-doc-classification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-classification-doc-classification"}},[e._v("#")]),e._v(" biome.text.modules.heads.classification.doc_classification "),a("Badge",{attrs:{text:"Module"}})],1),e._v(" "),a("div"),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"documentclassification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#documentclassification"}},[e._v("#")]),e._v(" DocumentClassification "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("DocumentClassification")]),e._v(" ("),e._v("\n    "),a("span",[e._v("backbone: "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"../../../backbone.html#biome.text.backbone.ModelBackbone"}},[e._v("ModelBackbone")])]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("labels: List[str]")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("token_pooler: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration"}},[e._v("Seq2VecEncoderConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("sentence_encoder: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration"}},[e._v("Seq2SeqEncoderConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("sentence_pooler: "),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration"}},[e._v("Seq2VecEncoderConfiguration")]),e._v(" = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("feedforward: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration"}},[e._v("FeedForwardConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("dropout: float = 0.0")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("multilabel: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("label_weights: Union[List[float], Dict[str, float], NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Task head for document text classification. It's quite similar to text\nclassification but including the doc2vec transformation layers")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("backbone")])])]),e._v(" "),a("dd",[e._v("The backbone of your model. Must not be provided when initiating with "),a("code",[e._v("Pipeline.from_config")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("labels")])])]),e._v(" "),a("dd",[e._v("A list of labels for your classification task.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("token_pooler")])])]),e._v(" "),a("dd",[e._v("The pooler at token level to provide one vector per sentence. Default: "),a("code",[e._v("BagOfEmbeddingsEncoder")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("sentence_encoder")])])]),e._v(" "),a("dd",[e._v("An optional sequence to sequence encoder that contextualizes the sentence representations. Default: None.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("sentence_pooler")])])]),e._v(" "),a("dd",[e._v("The pooler at sentence level to provide a vector for the document. Default: "),a("code",[e._v("BagOfEmbeddingsEncoder")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("feedforward")])])]),e._v(" "),a("dd",[e._v("An optional feedforward layer applied to the output of the sentence pooler. Default: None.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("dropout")])])]),e._v(" "),a("dd",[e._v("A dropout applied after the backbone, the token_pooler, the sentence_encoder and sentence_pooler. Default: 0.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("multilabel")])])]),e._v(" "),a("dd",[e._v("Is this a multi label classification task? Default: False")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("label_weights")])])]),e._v(" "),a("dd",[e._v("A list of weights for each label. The weights must be in the same order as the "),a("code",[e._v("labels")]),e._v(".\nYou can also provide a dictionary that maps the label to its weight. Default: None.")])]),e._v(" "),a("p",[e._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead"}},[e._v("ClassificationHead")])]),e._v(" "),a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[e._v("TaskHead")])]),e._v(" "),a("li",[e._v("torch.nn.modules.module.Module")]),e._v(" "),a("li",[e._v("allennlp.common.registrable.Registrable")]),e._v(" "),a("li",[e._v("allennlp.common.from_params.FromParams")]),e._v(" "),a("li",[e._v("allennlp.common.det_hash.CustomDetHash")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"subclasses"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#subclasses"}},[e._v("#")]),e._v(" Subclasses")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.classification.record_classification.RecordClassification",href:"record_classification.html#biome.text.modules.heads.classification.record_classification.RecordClassification"}},[e._v("RecordClassification")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members"}},[e._v("#")]),e._v(" Inherited members")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead"}},[e._v("ClassificationHead")])])]),e._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.empty_prediction",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.empty_prediction"}},[e._v("empty_prediction")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.extend_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels"}},[e._v("extend_labels")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.featurize",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.featurize"}},[e._v("featurize")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.forward",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.forward"}},[e._v("forward")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics"}},[e._v("get_metrics")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.inputs",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.inputs"}},[e._v("inputs")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.labels"}},[e._v("labels")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.make_task_prediction",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.make_task_prediction"}},[e._v("make_task_prediction")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.num_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels"}},[e._v("num_labels")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.on_vocab_update",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update"}},[e._v("on_vocab_update")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.register",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.register"}},[e._v("register")])])])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"documentclassificationconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#documentclassificationconfiguration"}},[e._v("#")]),e._v(" DocumentClassificationConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("DocumentClassificationConfiguration")]),e._v(" (*args, **kwds)"),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Lazy initialization for document classification head components")]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[e._v("ComponentConfiguration")])]),e._v(" "),a("li",[e._v("typing.Generic")]),e._v(" "),a("li",[e._v("allennlp.common.from_params.FromParams")]),e._v(" "),a("li",[e._v("allennlp.common.det_hash.CustomDetHash")])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members-2"}},[e._v("#")]),e._v(" Inherited members")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[e._v("ComponentConfiguration")])])]),e._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.compile",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.compile"}},[e._v("compile")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.config",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.config"}},[e._v("config")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.from_params",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.from_params"}},[e._v("from_params")])])]),e._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.input_dim",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.input_dim"}},[e._v("input_dim")])])])])])])])}),[],!1,null,null,null);t.default=o.exports}}]);