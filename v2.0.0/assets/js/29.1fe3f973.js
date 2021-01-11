(window.webpackJsonp=window.webpackJsonp||[]).push([[29],{430:function(t,e,a){"use strict";a.r(e);var s=a(26),n=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-explore"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-explore"}},[t._v("#")]),t._v(" biome.text.explore "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"create"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#create"}},[t._v("#")]),t._v(" create "),a("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("create")]),t._v(" ("),t._v("\n  pipeline: "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"pipeline.html#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")]),t._v(",\n  dataset: "),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[t._v("Dataset")]),t._v(",\n  explore_id: Union[str, NoneType] = None,\n  es_host: Union[str, NoneType] = None,\n  batch_size: int = 50,\n  num_proc: int = 1,\n  prediction_cache_size: int = 0,\n  explain: bool = False,\n  force_delete: bool = True,\n  show_explore: bool = True,\n  **metadata,\n)  -> str\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Launches the Explore UI for a given data source")]),t._v(" "),a("p",[t._v("Running this method inside an "),a("code",[t._v("IPython")]),t._v(" notebook will try to render the UI directly in the notebook.")]),t._v(" "),a("p",[t._v("Running this outside a notebook will try to launch the standalone web application.")]),t._v(" "),a("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("pipeline")])])]),t._v(" "),a("dd",[t._v("Pipeline used for data exploration")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("dataset")])])]),t._v(" "),a("dd",[t._v("The dataset to explore")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("explore_id")])])]),t._v(" "),a("dd",[t._v("A name or id for this explore run, useful for running and keep track of several explorations")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("es_host")])])]),t._v(" "),a("dd",[t._v("The URL to the Elasticsearch host for indexing predictions (default is "),a("code",[t._v("localhost:9200")]),t._v(")")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("batch_size")])])]),t._v(" "),a("dd",[t._v("Batch size for the predictions")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("num_proc")])])]),t._v(" "),a("dd",[t._v("Only for Dataset: Number of processes to run predictions in parallel (default: 1)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("prediction_cache_size")])])]),t._v(" "),a("dd",[t._v("The size of the cache for caching predictions (default is `0)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("explain")])])]),t._v(" "),a("dd",[t._v("Whether to extract and return explanations of token importance (default is "),a("code",[t._v("False")]),t._v(")")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("force_delete")])])]),t._v(" "),a("dd",[t._v("Deletes exploration with the same "),a("code",[t._v("explore_id")]),t._v(" before indexing the new explore items (default is `True)")]),t._v(" "),a("dt",[a("strong",[a("code",[t._v("show_explore")])])]),t._v(" "),a("dd",[t._v("If true, show ui for data exploration interaction (default is "),a("code",[t._v("True")]),t._v(")")])])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"show"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#show"}},[t._v("#")]),t._v(" show "),a("Badge",{attrs:{text:"Function"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("show")]),t._v(" ("),t._v("\n  explore_id: str,\n  es_host: Union[str, NoneType] = None,\n)  -> NoneType\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Shows explore ui for data prediction exploration")])]),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"dataexploration"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#dataexploration"}},[t._v("#")]),t._v(" DataExploration "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("DataExploration")]),t._v(" ("),t._v("\n    "),a("span",[t._v("name: str")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("pipeline: "),a("a",{attrs:{title:"biome.text.pipeline.Pipeline",href:"pipeline.html#biome.text.pipeline.Pipeline"}},[t._v("Pipeline")])]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("use_prediction: bool")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("dataset_name: str")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("dataset_columns: List[str] = <factory>")]),a("span",[t._v(",")]),t._v("\n    "),a("span",[t._v("metadata: Dict[str, Any] = <factory>")]),a("span",[t._v(",")]),t._v("\n"),a("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("Data exploration info")]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"as-old-format"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#as-old-format"}},[t._v("#")]),t._v(" as_old_format "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("as_old_format")]),t._v("("),a("span",[t._v("self) -> Dict[str, Any]")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("h2",{attrs:{id:"returns"}},[t._v("Returns")])])])])}),[],!1,null,null,null);e.default=n.exports}}]);
