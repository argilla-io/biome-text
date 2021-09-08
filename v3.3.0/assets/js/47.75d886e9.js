(window.webpackJsonp=window.webpackJsonp||[]).push([[47],{464:function(t,e,a){"use strict";a.r(e);var s=a(28),o=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-modules-heads-token-classification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-token-classification"}},[t._v("#")]),t._v(" biome.text.modules.heads.token_classification "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"tokenclassification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tokenclassification"}},[t._v("#")]),t._v(" TokenClassification "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TokenClassification")]),t._v(" ("),t._v("\n    "),a("span",[t._v("backbone: "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"../../backbone.html#biome.text.backbone.ModelBackbone"}},[t._v("ModelBackbone")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("labels: List[str]")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("label_encoding: Union[str, NoneType] = 'BIOUL'")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("top_k: int = 1")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("dropout: Union[float, NoneType] = 0.0")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("feedforward: Union["),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration",href:"../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration"}},[t._v("FeedForwardConfiguration")]),t._v(", NoneType] = None")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Task head for token classification (NER)")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("backbone")])])]),t._v(" "),a("dd",[t._v("The model backbone")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("labels")])])]),t._v(" "),a("dd",[t._v("List span labels. Span labels get converted to tag labels internally, using\nconfigured label_encoding for that.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("label_encoding")])])]),t._v(" "),a("dd",[t._v("The format of the tags. Supported encodings are: ['BIO', 'BIOUL']")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("top_k")])])]),t._v(" "),a("dd",[t._v(" ")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("dropout")])])]),t._v(" "),a("dd",[t._v(" ")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("feedforward")])])]),t._v(" "),a("dd",[t._v(" ")])]),t._v(" "),a("p",[t._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")])]),t._v(" "),a("li",[t._v("torch.nn.modules.module.Module")]),t._v(" "),a("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")]),t._v(" "),a("li",[t._v("allennlp.common.det_hash.CustomDetHash")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"instance-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[t._v("#")]),t._v(" Instance variables")]),t._v("\n")]),t._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.modules.heads.token_classification.TokenClassification.span_labels"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("span_labels")]),t._v(" : List[str]")])]),t._v(" "),a("dd")]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"featurize"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#featurize"}},[t._v("#")]),t._v(" featurize "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("featurize")]),t._v(" ("),t._v("\n  self,\n  text: Union[str, List[str]],\n  entities: Union[List[dict], NoneType] = None,\n  tags: Union[List[str], List[int], NoneType] = None,\n)  -> allennlp.data.instance.Instance\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("text")])])]),t._v(" "),a("dd",[t._v("Can be either a simple str or a list of str,\nin which case it will be treated as a list of pretokenized tokens")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("entities")])])]),t._v(" "),a("dd",[a("p",[t._v("A list of span labels")]),t._v(" "),a("p",[t._v("Span labels are dictionaries that contain:")]),t._v(" "),a("p",[t._v("'start': int, char index of the start of the span\n'end': int, char index of the end of the span (exclusive)\n'label': str, label of the span")]),t._v(" "),a("p",[t._v("They are used with the "),a("code",[t._v("spacy.gold.biluo_tags_from_offsets")]),t._v(" method.")])]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("tags")])])]),t._v(" "),a("dd",[t._v("A list of tags in the BIOUL or BIO format.")])])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members"}},[t._v("#")]),t._v(" Inherited members")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")])])]),t._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.empty_prediction",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.empty_prediction"}},[t._v("empty_prediction")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.extend_labels",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels"}},[t._v("extend_labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.forward",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.forward"}},[t._v("forward")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.get_metrics",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.get_metrics"}},[t._v("get_metrics")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.inputs",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.inputs"}},[t._v("inputs")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.labels",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.labels"}},[t._v("labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.make_task_prediction",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.make_task_prediction"}},[t._v("make_task_prediction")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.num_labels",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels"}},[t._v("num_labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.on_vocab_update",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update"}},[t._v("on_vocab_update")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead.register",href:"task_head.html#biome.text.modules.heads.task_head.TaskHead.register"}},[t._v("register")])])])])])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"tokenclassificationconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tokenclassificationconfiguration"}},[t._v("#")]),t._v(" TokenClassificationConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TokenClassificationConfiguration")]),t._v(" (*args, **kwds)"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Configuration for classification head components")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[t._v("ComponentConfiguration")])]),t._v(" "),a("li",[t._v("typing.Generic")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")]),t._v(" "),a("li",[t._v("allennlp.common.det_hash.CustomDetHash")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members-2"}},[t._v("#")]),t._v(" Inherited members")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[t._v("ComponentConfiguration")])])]),t._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.compile",href:"../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.compile"}},[t._v("compile")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.config",href:"../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.config"}},[t._v("config")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.from_params",href:"../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.from_params"}},[t._v("from_params")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.input_dim",href:"../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.input_dim"}},[t._v("input_dim")])])])])])])])}),[],!1,null,null,null);e.default=o.exports}}]);