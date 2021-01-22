(window.webpackJsonp=window.webpackJsonp||[]).push([[49],{460:function(t,s,a){"use strict";a.r(s);var e=a(26),i=Object(e.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-modules-heads-task-prediction"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-task-prediction"}},[t._v("#")]),t._v(" biome.text.modules.heads.task_prediction "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"token"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#token"}},[t._v("#")]),t._v(" Token "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("Token")]),t._v(" ("),t._v("\n    "),a("span",[t._v("text: str")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("start: int")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("end: int")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Output dataclass for a token in a prediction.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("text")])])]),t._v(" "),a("dd",[t._v("Text of the token")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("start")])])]),t._v(" "),a("dd",[t._v("Start char id")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("end")])])]),t._v(" "),a("dd",[t._v("End char id")])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"entity"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#entity"}},[t._v("#")]),t._v(" Entity "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("Entity")]),t._v(" ("),t._v("\n    "),a("span",[t._v("label: str")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("start_token: int")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("end_token: int")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("start: Union[int, NoneType] = 'SENTINEL TO SKIP DATACLASS FIELDS WHEN CONVERTING TO DICT'")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("end: Union[int, NoneType] = 'SENTINEL TO SKIP DATACLASS FIELDS WHEN CONVERTING TO DICT'")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Output dataclass for a NER entity in a prediction.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("label")])])]),t._v(" "),a("dd",[t._v("Label of the entity")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("start_token")])])]),t._v(" "),a("dd",[t._v("Start token id")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("end_token")])])]),t._v(" "),a("dd",[t._v("End token id")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("start")])])]),t._v(" "),a("dd",[t._v("Start char id")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("end")])])]),t._v(" "),a("dd",[t._v("End char id")])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"taskprediction"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#taskprediction"}},[t._v("#")]),t._v(" TaskPrediction "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TaskPrediction")]),t._v(" ()"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Base class for the TaskOutput classes.")]),t._v(" "),a("p",[t._v("Each head should implement a proper task prediction class that defines its prediction output.\nYou can use the SENTINEL as default value if you want to omit certain fields when converting to a dict.")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"subclasses"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#subclasses"}},[t._v("#")]),t._v(" Subclasses")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.ClassificationPrediction",href:"#biome.text.modules.heads.task_prediction.ClassificationPrediction"}},[t._v("ClassificationPrediction")])]),t._v(" "),a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.LanguageModellingPrediction",href:"#biome.text.modules.heads.task_prediction.LanguageModellingPrediction"}},[t._v("LanguageModellingPrediction")])]),t._v(" "),a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.TokenClassificationPrediction",href:"#biome.text.modules.heads.task_prediction.TokenClassificationPrediction"}},[t._v("TokenClassificationPrediction")])])]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"as-dict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#as-dict"}},[t._v("#")]),t._v(" as_dict "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("as_dict")]),t._v("("),a("span",[t._v("self) -> Dict")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd")]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"classificationprediction"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#classificationprediction"}},[t._v("#")]),t._v(" ClassificationPrediction "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("ClassificationPrediction")]),t._v(" (labels: List[str], probabilities: List[float])"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Output dataclass for all "),a("code",[t._v("ClassificationHead")]),t._v("s:\n- "),a("code",[t._v("TextClassification")]),t._v("\n- "),a("code",[t._v("RecordClassification")]),t._v("\n- "),a("code",[t._v("DocumentClassification")]),t._v("\n- "),a("code",[t._v("RecordPairClassification")])]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("labels")])])]),t._v(" "),a("dd",[t._v("Ordered list of predictions, from the label with the highest to the label with the lowest probability.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("probabilities")])])]),t._v(" "),a("dd",[t._v("Ordered list of probabilities, from highest to lowest probability.")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.TaskPrediction",href:"#biome.text.modules.heads.task_prediction.TaskPrediction"}},[t._v("TaskPrediction")])])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"tokenclassificationprediction"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tokenclassificationprediction"}},[t._v("#")]),t._v(" TokenClassificationPrediction "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TokenClassificationPrediction")]),t._v(" ("),t._v("\n    "),a("span",[t._v("tags: List[List[str]]")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("entities: List[List["),a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.Entity",href:"#biome.text.modules.heads.task_prediction.Entity"}},[t._v("Entity")]),t._v("]]")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("scores: List[float]")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("tokens: Union[List["),a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.Token",href:"#biome.text.modules.heads.task_prediction.Token"}},[t._v("Token")]),t._v("], NoneType] = 'SENTINEL TO SKIP DATACLASS FIELDS WHEN CONVERTING TO DICT'")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Output dataclass for the "),a("code",[t._v("TokenClassification")]),t._v(" head")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("tags")])])]),t._v(" "),a("dd",[t._v("List of lists of NER tags, ordered by score.\nThe list of NER tags with the highest score comes first.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("entities")])])]),t._v(" "),a("dd",[t._v("List of list of entities, ordered by score.\nThe list of entities with the highest score comes first.")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("scores")])])]),t._v(" "),a("dd",[t._v("Ordered list of scores for each list of NER tags (highest to lowest).")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("tokens")])])]),t._v(" "),a("dd",[t._v("Tokens of the tokenized input")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.TaskPrediction",href:"#biome.text.modules.heads.task_prediction.TaskPrediction"}},[t._v("TaskPrediction")])])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"languagemodellingprediction"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#languagemodellingprediction"}},[t._v("#")]),t._v(" LanguageModellingPrediction "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("LanguageModellingPrediction")]),t._v(" ("),t._v("\n    "),a("span",[t._v("lm_embeddings: "),a("built-in",{attrs:{function:"",array:""}})],1),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("mask: "),a("built-in",{attrs:{function:"",array:""}})],1),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("loss: Union[float, NoneType] = 'SENTINEL TO SKIP DATACLASS FIELDS WHEN CONVERTING TO DICT'")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Output dataclass for the "),a("code",[t._v("LanguageModelling")]),t._v(" head")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-3"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.modules.heads.task_prediction.TaskPrediction",href:"#biome.text.modules.heads.task_prediction.TaskPrediction"}},[t._v("TaskPrediction")])])])])}),[],!1,null,null,null);s.default=i.exports}}]);