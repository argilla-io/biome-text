(window.webpackJsonp=window.webpackJsonp||[]).push([[25],{433:function(t,a,s){"use strict";s.r(a);var e=s(27),n=Object(e.a)({},(function(){var t=this,a=t.$createElement,s=t._self._c||a;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h1",{attrs:{id:"biome-text-helpers"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-helpers"}},[t._v("#")]),t._v(" biome.text.helpers "),s("Badge",{attrs:{text:"Module"}})],1),t._v(" "),s("div"),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"yaml-to-dict"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#yaml-to-dict"}},[t._v("#")]),t._v(" yaml_to_dict "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("yaml_to_dict")]),t._v("("),s("span",[t._v("filepath: str) -> Dict[str, Any]")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Loads a yaml file into a data dictionary")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("filepath")])])]),t._v(" "),s("dd",[t._v("Path to the yaml file")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("dict")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"update-method-signature"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#update-method-signature"}},[t._v("#")]),t._v(" update_method_signature "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("update_method_signature")]),t._v(" ("),t._v("\n  signature: inspect.Signature,\n  to_method: Callable,\n)  -> Callable\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Updates the signature of a method")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("signature")])])]),t._v(" "),s("dd",[t._v("The signature with which to update the method")]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("to_method")])])]),t._v(" "),s("dd",[t._v("The method whose signature will be updated")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("updated_method")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"isgeneric"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#isgeneric"}},[t._v("#")]),t._v(" isgeneric "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("isgeneric")]),t._v("("),s("span",[t._v("class_type: Type) -> bool")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Checks if a class type is a generic type (List[str] or Union[str, int]")])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"is-running-on-notebook"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#is-running-on-notebook"}},[t._v("#")]),t._v(" is_running_on_notebook "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("is_running_on_notebook")]),t._v("("),s("span",[t._v(") -> bool")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Checks if code is running inside a jupyter notebook")])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"split-signature-params-by-predicate"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#split-signature-params-by-predicate"}},[t._v("#")]),t._v(" split_signature_params_by_predicate "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("split_signature_params_by_predicate")]),t._v(" ("),t._v("\n  signature_function: Callable,\n  predicate: Callable,\n)  -> Tuple[List[inspect.Parameter], List[inspect.Parameter]]\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Splits parameters signature by defined boolean predicate function")])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"sanitize-metric-name"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sanitize-metric-name"}},[t._v("#")]),t._v(" sanitize_metric_name "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("sanitize_metric_name")]),t._v("("),s("span",[t._v("name: str) -> str")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Sanitizes the name to comply with tensorboardX conventions when logging.")]),t._v(" "),s("h2",{attrs:{id:"parameter"}},[t._v("Parameter")]),t._v(" "),s("p",[t._v("name\nName of the metric")]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("sanitized_name")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"save-dict-as-yaml"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#save-dict-as-yaml"}},[t._v("#")]),t._v(" save_dict_as_yaml "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("save_dict_as_yaml")]),t._v(" ("),t._v("\n  dictionary: dict,\n  path: str,\n)  -> str\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Save a cfg dict to path as yaml")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("dictionary")])])]),t._v(" "),s("dd",[t._v("Dictionary to be saved")]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("path")])])]),t._v(" "),s("dd",[t._v("Filesystem location where the yaml file will be saved")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("path")])]),t._v(" "),s("dd",[t._v("Location of the yaml file")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"get-full-class-name"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-full-class-name"}},[t._v("#")]),t._v(" get_full_class_name "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("get_full_class_name")]),t._v("("),s("span",[t._v("the_class: Type) -> str")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Given a type class return the full qualified class name")])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"stringify"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#stringify"}},[t._v("#")]),t._v(" stringify "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("stringify")]),t._v("("),s("span",[t._v("value: Any) -> Any")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Creates an equivalent data structure representing data values as string")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("value")])])]),t._v(" "),s("dd",[t._v("Value to be stringified")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("stringified_value")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"sanitize-for-params"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sanitize-for-params"}},[t._v("#")]),t._v(" sanitize_for_params "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("sanitize_for_params")]),t._v("("),s("span",[t._v("x: Any) -> Any")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Sanitizes the input for a more flexible usage with AllenNLP's "),s("code",[t._v(".from_params()")]),t._v(" machinery.")]),t._v(" "),s("p",[t._v("For now it is mainly used to transform numpy numbers to python types")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("x")])])]),t._v(" "),s("dd",[t._v("The parameter passed on to "),s("code",[t._v("allennlp.common.FromParams.from_params()")])])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("sanitized_x")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"sanitize-for-yaml"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sanitize-for-yaml"}},[t._v("#")]),t._v(" sanitize_for_yaml "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("sanitize_for_yaml")]),t._v("("),s("span",[t._v("value: Any)")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Sanitizes the value for a simple yaml output, that is classes only built-in types")])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"span-labels-to-tag-labels"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#span-labels-to-tag-labels"}},[t._v("#")]),t._v(" span_labels_to_tag_labels "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("span_labels_to_tag_labels")]),t._v(" ("),t._v("\n  labels: List[str],\n  label_encoding: str = 'BIO',\n)  -> List[str]\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Converts a list of span labels to tag labels following "),s("code",[t._v("spacy.training.offsets_to_biluo_tags")])]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("labels")])])]),t._v(" "),s("dd",[t._v("Span labels to convert")]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("label_encoding")])])]),t._v(" "),s("dd",[t._v("The label format used for the tag labels")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("tag_labels")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"bioul-tags-to-bio-tags"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#bioul-tags-to-bio-tags"}},[t._v("#")]),t._v(" bioul_tags_to_bio_tags "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("bioul_tags_to_bio_tags")]),t._v("("),s("span",[t._v("tags: List[str]) -> List[str]")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Converts BIOUL tags to BIO tags")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("tags")])])]),t._v(" "),s("dd",[t._v("BIOUL tags to convert")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("bio_tags")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"tags-from-offsets"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tags-from-offsets"}},[t._v("#")]),t._v(" tags_from_offsets "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("tags_from_offsets")]),t._v(" ("),t._v("\n  doc: spacy.tokens.doc.Doc,\n  offsets: List[Dict],\n  label_encoding: Union[str, NoneType] = 'BIOUL',\n)  -> List[str]\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Converts offsets to BIOUL or BIO tags using spacy's "),s("code",[t._v("offsets_to_biluo_tags")]),t._v(".")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("doc")])])]),t._v(" "),s("dd",[t._v("A spaCy Doc created with "),s("code",[t._v("text")]),t._v(" and the backbone tokenizer")]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("offsets")])])]),t._v(" "),s("dd",[t._v("A list of dicts with start and end character index with respect to the doc, and the span label:\n"),s("code",[t._v('{"start": int, "end": int, "label": str}')])]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("label_encoding")])])]),t._v(" "),s("dd",[t._v("The label encoding to be used: BIOUL or BIO")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("tags (BIOUL")]),t._v(" or "),s("code",[t._v("BIO)")])]),t._v(" "),s("dd",[t._v(" ")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"offsets-from-tags"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#offsets-from-tags"}},[t._v("#")]),t._v(" offsets_from_tags "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("offsets_from_tags")]),t._v(" ("),t._v("\n  doc: spacy.tokens.doc.Doc,\n  tags: List[str],\n  label_encoding: Union[str, NoneType] = 'BIOUL',\n  only_token_spans: bool = False,\n)  -> List[Dict]\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Converts BIOUL or BIO tags to offsets")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("doc")])])]),t._v(" "),s("dd",[t._v("A spaCy Doc created with "),s("code",[t._v("text")]),t._v(" and the backbone tokenizer")]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("tags")])])]),t._v(" "),s("dd",[t._v("A list of BIOUL or BIO tags")]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("label_encoding")])])]),t._v(" "),s("dd",[t._v("The label encoding of the tags: BIOUL or BIO")]),t._v(" "),s("dt",[s("strong",[s("code",[t._v("only_token_spans")])])]),t._v(" "),s("dd",[t._v("If True, offsets contains only token index references. Default is False")])]),t._v(" "),s("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),s("dl",[s("dt",[s("code",[t._v("offsets")])]),t._v(" "),s("dd",[t._v("A list of dicts with start and end character/token index with respect to the doc and the span label:\n"),s("code",[t._v('{"start": int, "end": int, "start_token": int, "end_token": int, "label": str}')])])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"merge-dicts"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#merge-dicts"}},[t._v("#")]),t._v(" merge_dicts "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("merge_dicts")]),t._v(" ("),t._v("\n  source: Dict[str, Any],\n  destination: Dict[str, Any],\n)  -> Dict[str, Any]\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Merge two dictionaries recursivelly")]),t._v(" "),s("h2",{attrs:{id:"examples"}},[t._v("Examples")]),t._v(" "),s("pre",[s("code",{staticClass:"language-python"},[t._v(">>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }\n>>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }\n>>> merge_dicts(b, a)\n{'first': {'all_rows': {'pass': 'dog', 'number': '5', 'fail': 'cat'}}}\n")])])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"copy-sign-and-docs"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#copy-sign-and-docs"}},[t._v("#")]),t._v(" copy_sign_and_docs "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("copy_sign_and_docs")]),t._v("("),s("span",[t._v("org_func)")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Decorator to copy the signature and the docstring from the org_func")])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"spacy-to-allennlp-token"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#spacy-to-allennlp-token"}},[t._v("#")]),t._v(" spacy_to_allennlp_token "),s("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("spacy_to_allennlp_token")]),t._v("("),s("span",[t._v("token: spacy.tokens.token.Token) -> allennlp.data.tokenizers.token_class.Token")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd")])}),[],!1,null,null,null);a.default=n.exports}}]);