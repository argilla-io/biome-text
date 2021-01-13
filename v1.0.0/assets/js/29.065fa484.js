(window.webpackJsonp=window.webpackJsonp||[]).push([[29],{440:function(t,a,e){"use strict";e.r(a);var s=e(26),n=Object(s.a)({},(function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"biome-text-features"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-features"}},[t._v("#")]),t._v(" biome.text.features "),e("Badge",{attrs:{text:"Module"}})],1),t._v(" "),e("div"),t._v(" "),e("div"),t._v(" "),e("pre",{staticClass:"title"},[e("h2",{attrs:{id:"wordfeatures"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#wordfeatures"}},[t._v("#")]),t._v(" WordFeatures "),e("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("WordFeatures")]),t._v(" ("),t._v("\n    "),e("span",[t._v("embedding_dim: int")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("lowercase_tokens: bool = False")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("trainable: bool = True")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("weights_file: Union[str, NoneType] = None")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("**extra_params")]),e("span",[t._v(",")]),t._v("\n"),e("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),e("p",[t._v("Feature configuration at word level")]),t._v(" "),e("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),e("dl",[e("dt",[e("strong",[e("code",[t._v("embedding_dim")])])]),t._v(" "),e("dd",[t._v("Dimension of the embeddings")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("lowercase_tokens")])])]),t._v(" "),e("dd",[t._v("If True, lowercase tokens before the indexing")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("trainable")])])]),t._v(" "),e("dd",[t._v("If False, freeze the embeddings")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("weights_file")])])]),t._v(" "),e("dd",[t._v("Path to a file with pretrained weights for the embedding")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("**extra_params")])])]),t._v(" "),e("dd",[t._v("Extra parameters passed on to the "),e("code",[t._v("indexer")]),t._v(" and "),e("code",[t._v("embedder")]),t._v(" of the AllenNLP configuration framework.\nFor example: "),e("code",[t._v('WordFeatures(embedding_dim=300, embedder={"padding_index": 0})')])])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"class-variables"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#class-variables"}},[t._v("#")]),t._v(" Class variables")]),t._v("\n")]),t._v(" "),e("dl",[e("dt",{attrs:{id:"biome.text.features.WordFeatures.namespace"}},[e("code",{staticClass:"name"},[t._v("var "),e("span",{staticClass:"ident"},[t._v("namespace")])])]),t._v(" "),e("dd")]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"instance-variables"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables"}},[t._v("#")]),t._v(" Instance variables")]),t._v("\n")]),t._v(" "),e("dl",[e("dt",{attrs:{id:"biome.text.features.WordFeatures.config"}},[e("code",{staticClass:"name"},[t._v("var "),e("span",{staticClass:"ident"},[t._v("config")]),t._v(" : Dict")])]),t._v(" "),e("dd",[e("p",[t._v("Returns the config in AllenNLP format")])])]),t._v(" "),e("dl",[e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"to-json"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#to-json"}},[t._v("#")]),t._v(" to_json "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("to_json")]),t._v("("),e("span",[t._v("self) -> Dict")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Returns the config as dict for the serialized json config file")])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"to-dict"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#to-dict"}},[t._v("#")]),t._v(" to_dict "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("to_dict")]),t._v("("),e("span",[t._v("self) -> Dict")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Returns the config as dict")])])]),t._v(" "),e("div"),t._v(" "),e("pre",{staticClass:"title"},[e("h2",{attrs:{id:"charfeatures"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#charfeatures"}},[t._v("#")]),t._v(" CharFeatures "),e("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("CharFeatures")]),t._v(" ("),t._v("\n    "),e("span",[t._v("embedding_dim: int")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("encoder: Dict[str, Any]")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("dropout: float = 0.0")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("lowercase_characters: bool = False")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("**extra_params")]),e("span",[t._v(",")]),t._v("\n"),e("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),e("p",[t._v("Feature configuration at character level")]),t._v(" "),e("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),e("dl",[e("dt",[e("strong",[e("code",[t._v("embedding_dim")])])]),t._v(" "),e("dd",[t._v("Dimension of the character embeddings.")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("encoder")])])]),t._v(" "),e("dd",[t._v("A sequence to vector encoder resulting in a word representation based on its characters")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("dropout")])])]),t._v(" "),e("dd",[t._v("Dropout applied to the output of the encoder")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("lowercase_characters")])])]),t._v(" "),e("dd",[t._v("If True, lowercase characters before the indexing")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("**extra_params")])])]),t._v(" "),e("dd",[t._v("Extra parameters passed on to the "),e("code",[t._v("indexer")]),t._v(" and "),e("code",[t._v("embedder")]),t._v(" of the AllenNLP configuration framework.\nFor example: "),e("code",[t._v('CharFeatures(embedding_dim=32, indexer={"min_padding_length": 5}, ...)')])])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"class-variables-2"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#class-variables-2"}},[t._v("#")]),t._v(" Class variables")]),t._v("\n")]),t._v(" "),e("dl",[e("dt",{attrs:{id:"biome.text.features.CharFeatures.namespace"}},[e("code",{staticClass:"name"},[t._v("var "),e("span",{staticClass:"ident"},[t._v("namespace")])])]),t._v(" "),e("dd")]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"instance-variables-2"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#instance-variables-2"}},[t._v("#")]),t._v(" Instance variables")]),t._v("\n")]),t._v(" "),e("dl",[e("dt",{attrs:{id:"biome.text.features.CharFeatures.config"}},[e("code",{staticClass:"name"},[t._v("var "),e("span",{staticClass:"ident"},[t._v("config")]),t._v(" : Dict")])]),t._v(" "),e("dd",[e("p",[t._v("Returns the config in AllenNLP format")])])]),t._v(" "),e("dl",[e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"to-json-2"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#to-json-2"}},[t._v("#")]),t._v(" to_json "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("to_json")]),t._v("("),e("span",[t._v("self)")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Returns the config as dict for the serialized json config file")])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"to-dict-2"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#to-dict-2"}},[t._v("#")]),t._v(" to_dict "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("to_dict")]),t._v("("),e("span",[t._v("self)")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Returns the config as dict")])])])])}),[],!1,null,null,null);a.default=n.exports}}]);