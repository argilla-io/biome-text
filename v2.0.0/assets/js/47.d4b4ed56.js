(window.webpackJsonp=window.webpackJsonp||[]).push([[47],{448:function(t,e,a){"use strict";a.r(e);var i=a(26),o=Object(i.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-modules-heads-classification-record-pair-classification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-classification-record-pair-classification"}},[t._v("#")]),t._v(" biome.text.modules.heads.classification.record_pair_classification "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"recordpairclassification"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#recordpairclassification"}},[t._v("#")]),t._v(" RecordPairClassification "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("RecordPairClassification")]),t._v(" ("),t._v("\n    "),a("span",[t._v("backbone: "),a("a",{attrs:{title:"biome.text.backbone.ModelBackbone",href:"../../../backbone.html#biome.text.backbone.ModelBackbone"}},[t._v("ModelBackbone")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("labels: List[str]")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("field_encoder: "),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration"}},[t._v("Seq2VecEncoderConfiguration")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("record_encoder: "),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration"}},[t._v("Seq2SeqEncoderConfiguration")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("matcher_forward: "),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration"}},[t._v("BiMpmMatchingConfiguration")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("aggregator: "),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration"}},[t._v("Seq2VecEncoderConfiguration")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("classifier_feedforward: "),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration"}},[t._v("FeedForwardConfiguration")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("matcher_backward: "),a("a",{attrs:{title:"biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration",href:"../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration"}},[t._v("BiMpmMatchingConfiguration")]),t._v(" = None")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("dropout: float = 0.1")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("initializer: allennlp.nn.initializers.InitializerApplicator = <allennlp.nn.initializers.InitializerApplicator object>")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Classifies the relation between a pair of records using a matching layer.")]),t._v(" "),a("p",[t._v("The input for models using this "),a("code",[t._v("TaskHead")]),t._v(" are two "),a("em",[t._v("records")]),t._v(" with one or more "),a("em",[t._v("data fields")]),t._v(" each, and a label\ndescribing their relationship.\nIf you would like a meaningful explanation of the model's prediction,\nboth records must consist of the same number of "),a("em",[t._v("data fields")]),t._v(" and hold them in the same order.")]),t._v(" "),a("p",[t._v("The architecture is loosely based on the AllenNLP implementation of the BiMPM model described in\n"),a("code",[t._v("Bilateral Multi-Perspective Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>")]),t._v("_\nby Zhiguo Wang et al., 2017, and was adapted to deal with record pairs.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("backbone")])])]),t._v(" "),a("dd",[t._v("Takes care of the embedding and optionally of the language encoding")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("labels")])])]),t._v(" "),a("dd",[t._v("List of labels")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("field_encoder")])])]),t._v(" "),a("dd",[t._v("Encodes a data field, contextualized within the field")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("record_encoder")])])]),t._v(" "),a("dd",[t._v("Encodes data fields, contextualized within the record")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("matcher_forward")])])]),t._v(" "),a("dd",[t._v("BiMPM matching for the forward output of the record encoder layer")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("matcher_backward")])])]),t._v(" "),a("dd",[t._v("BiMPM matching for the backward output of the record encoder layer")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("aggregator")])])]),t._v(" "),a("dd",[t._v("Aggregator of all BiMPM matching vectors")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("classifier_feedforward")])])]),t._v(" "),a("dd",[t._v("Fully connected layers for classification.\nA linear output layer with the number of labels at the end will be added automatically!!!")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("dropout")])])]),t._v(" "),a("dd",[t._v("Dropout percentage to use.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("initializer")])])]),t._v(" "),a("dd",[t._v("If provided, will be used to initialize the model parameters.")])]),t._v(" "),a("p",[t._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead"}},[t._v("ClassificationHead")])]),t._v(" "),a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskHead",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead"}},[t._v("TaskHead")])]),t._v(" "),a("li",[t._v("torch.nn.modules.module.Module")]),t._v(" "),a("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"featurize"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#featurize"}},[t._v("#")]),t._v(" featurize "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("featurize")]),t._v(" ("),t._v("\n  self,\n  record1: Dict[str, Any],\n  record2: Dict[str, Any],\n  label: Union[str, NoneType] = None,\n)  -> Union[allennlp.data.instance.Instance, NoneType]\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Tokenizes, indexes and embeds the two records and optionally adds the label")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("record1")])]),t._v(" : "),a("code",[t._v("Dict[str, Any]")])]),t._v(" "),a("dd",[t._v("First record")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("record2")])]),t._v(" : "),a("code",[t._v("Dict[str, Any]")])]),t._v(" "),a("dd",[t._v("Second record")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("label")])]),t._v(" : "),a("code",[t._v("Optional[str]")])]),t._v(" "),a("dd",[t._v("Classification label")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("instance")])]),t._v(" "),a("dd",[t._v("AllenNLP instance containing the two records plus optionally a label")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"forward"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#forward"}},[t._v("#")]),t._v(" forward "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("forward")]),t._v(" ("),t._v("\n  self,\n  record1: Dict[str, Dict[str, torch.Tensor]],\n  record2: Dict[str, Dict[str, torch.Tensor]],\n  label: torch.LongTensor = None,\n)  -> "),a("a",{attrs:{title:"biome.text.modules.heads.task_head.TaskOutput",href:"../task_head.html#biome.text.modules.heads.task_head.TaskOutput"}},[t._v("TaskOutput")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("record1")])])]),t._v(" "),a("dd",[t._v("Tokens of the first record.\nThe dictionary is the output of a "),a("code",[t._v("ListField.as_array()")]),t._v(". It gives names to the tensors created by\nthe "),a("code",[t._v("TokenIndexer")]),t._v("s.\nIn its most basic form, using a "),a("code",[t._v("SingleIdTokenIndexer")]),t._v(", the dictionary is composed of:\n"),a("code",[t._v('{"tokens": {"tokens": Tensor(batch_size, num_fields, num_tokens)}}')]),t._v(".\nThe dictionary is designed to be passed on directly to a "),a("code",[t._v("TextFieldEmbedder")]),t._v(", that has a\n"),a("code",[t._v("TokenEmbedder")]),t._v(" for each key in the dictionary (except you set "),a("code",[t._v("allow_unmatched_keys")]),t._v(" in the\n"),a("code",[t._v("TextFieldEmbedder")]),t._v(" to False) and knows how to combine different word/character representations into a\nsingle vector per token in your input.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("record2")])])]),t._v(" "),a("dd",[t._v("Tokens of the second record.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("label")])]),t._v(" : "),a("code",[t._v("torch.LongTensor")]),t._v(", optional "),a("code",[t._v("(default = None)")])]),t._v(" "),a("dd",[t._v("A torch tensor representing the sequence of integer gold class label of shape\n"),a("code",[t._v("(batch_size, num_classes)")]),t._v(".")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("An output dictionary consisting of:")])]),t._v(" "),a("dd",[t._v(" ")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("logits")])]),t._v(" : "),a("code",[t._v("torch.FloatTensor")])]),t._v(" "),a("dd",[t._v(" ")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("class_probabilities")])]),t._v(" : "),a("code",[t._v("torch.FloatTensor")])]),t._v(" "),a("dd",[t._v(" ")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("loss")])]),t._v(" : "),a("code",[t._v("torch.FloatTensor")]),t._v(", optional")]),t._v(" "),a("dd",[t._v("A scalar loss to be optimised.")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"explain-prediction"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#explain-prediction"}},[t._v("#")]),t._v(" explain_prediction "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("explain_prediction")]),t._v(" ("),t._v("\n  self,\n  prediction: Dict[str, "),a("built-in",{attrs:{function:"",array:""}},[t._v("],\n  instance: allennlp.data.instance.Instance,\n  n_steps: int,\n)  -> Dict[str, Any]\n")])],1),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Calculates attributions for each data field in the record by integrating the gradients.")]),t._v(" "),a("p",[t._v("IMPORTANT: The calculated attributions only make sense for a duplicate/not_duplicate binary classification task\nof the two records.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("prediction")])])]),t._v(" "),a("dd",[t._v(" ")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("instance")])])]),t._v(" "),a("dd",[t._v(" ")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("n_steps")])])]),t._v(" "),a("dd",[t._v(" ")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("prediction_dict")])]),t._v(" "),a("dd",[t._v('The prediction dictionary with a newly added "explain" key')])])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members"}},[t._v("#")]),t._v(" Inherited members")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead"}},[t._v("ClassificationHead")])])]),t._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.add_label",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.add_label"}},[t._v("add_label")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.decode",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.decode"}},[t._v("decode")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.extend_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels"}},[t._v("extend_labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics",href:"classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics"}},[t._v("get_metrics")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.inputs",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.inputs"}},[t._v("inputs")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.labels"}},[t._v("labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.num_labels",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels"}},[t._v("num_labels")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.on_vocab_update",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update"}},[t._v("on_vocab_update")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.heads.classification.classification.ClassificationHead.register",href:"../task_head.html#biome.text.modules.heads.task_head.TaskHead.register"}},[t._v("register")])])])])])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"recordpairclassificationconfiguration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#recordpairclassificationconfiguration"}},[t._v("#")]),t._v(" RecordPairClassificationConfiguration "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("RecordPairClassificationConfiguration")]),t._v(" (*args, **kwds)"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Config for record pair classification head component")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[t._v("ComponentConfiguration")])]),t._v(" "),a("li",[t._v("typing.Generic")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"inherited-members-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#inherited-members-2"}},[t._v("#")]),t._v(" Inherited members")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("b",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration"}},[t._v("ComponentConfiguration")])])]),t._v(":\n"),a("ul",{staticClass:"hlist"},[a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.compile",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.compile"}},[t._v("compile")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.config",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.config"}},[t._v("config")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.from_params",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.from_params"}},[t._v("from_params")])])]),t._v(" "),a("li",[a("code",[a("a",{attrs:{title:"biome.text.modules.configuration.defs.ComponentConfiguration.input_dim",href:"../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.input_dim"}},[t._v("input_dim")])])])])])])])}),[],!1,null,null,null);e.default=o.exports}}]);
