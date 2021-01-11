(window.webpackJsonp=window.webpackJsonp||[]).push([[50],{458:function(t,e,a){"use strict";a.r(e);var s=a(26),n=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-text-cleaning"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-text-cleaning"}},[t._v("#")]),t._v(" biome.text.text_cleaning "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"textcleaning"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#textcleaning"}},[t._v("#")]),t._v(" TextCleaning "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TextCleaning")]),t._v(" ()"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Base class for text cleaning processors")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"subclasses"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#subclasses"}},[t._v("#")]),t._v(" Subclasses")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.text_cleaning.DefaultTextCleaning",href:"#biome.text.text_cleaning.DefaultTextCleaning"}},[t._v("DefaultTextCleaning")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"class-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#class-variables"}},[t._v("#")]),t._v(" Class variables")]),t._v("\n")]),t._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.text_cleaning.TextCleaning.default_implementation"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("default_implementation")]),t._v(" : str")])]),t._v(" "),a("dd")]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"textcleaningrule"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#textcleaningrule"}},[t._v("#")]),t._v(" TextCleaningRule "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TextCleaningRule")]),t._v(" (func: Callable[[str], str])"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Registers a function as a rule for the default text cleaning implementation")]),t._v(" "),a("p",[t._v("Use the decorator "),a("code",[t._v("@TextCleaningRule")]),t._v(" for creating custom text cleaning and pre-processing rules.")]),t._v(" "),a("p",[t._v("An example function to strip spaces (already included in the default "),a("code",[a("a",{attrs:{title:"biome.text.text_cleaning.TextCleaning",href:"#biome.text.text_cleaning.TextCleaning"}},[t._v("TextCleaning")])]),t._v(" processor):")]),t._v(" "),a("pre",[a("code",{staticClass:"language-python"},[t._v("@TextCleaningRule\ndef strip_spaces(text: str) -> str:\n    return text.strip()\n")])]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("func")])]),t._v(" : "),a("code",[t._v("Callable[[str]")])]),t._v(" "),a("dd",[t._v("The function to register")])]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"registered-rules"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#registered-rules"}},[t._v("#")]),t._v(" registered_rules "),a("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("registered_rules")]),t._v("("),a("span",[t._v(") -> Dict[str, Callable[[str], str]]")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Registered rules dictionary")])])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"defaulttextcleaning"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#defaulttextcleaning"}},[t._v("#")]),t._v(" DefaultTextCleaning "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("DefaultTextCleaning")]),t._v(" (rules: List[str] = None)"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Defines rules that can be applied to the text before it gets tokenized.")]),t._v(" "),a("p",[t._v("Each rule is a simple python function that receives and returns a "),a("code",[t._v("str")]),t._v(".")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("rules")])]),t._v(" : "),a("code",[t._v("List[str]")])]),t._v(" "),a("dd",[t._v("A list of registered rule method names to be applied to text inputs")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors-2"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[a("a",{attrs:{title:"biome.text.text_cleaning.TextCleaning",href:"#biome.text.text_cleaning.TextCleaning"}},[t._v("TextCleaning")])]),t._v(" "),a("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])])])}),[],!1,null,null,null);e.default=n.exports}}]);