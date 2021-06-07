(window.webpackJsonp=window.webpackJsonp||[]).push([[28],{434:function(t,e,a){"use strict";a.r(e);var s=a(26),o=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-mlflow-model"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-mlflow-model"}},[t._v("#")]),t._v(" biome.text.mlflow_model "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("div"),t._v(" "),a("div"),t._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"biometextmodel"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biometextmodel"}},[t._v("#")]),t._v(" BiomeTextModel "),a("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("BiomeTextModel")]),t._v(" ()"),t._v("\n")]),t._v("\n")]),t._v(" "),a("p",[t._v("A custom MLflow model with the 'python_function' flavor for biome.text pipelines.")]),t._v(" "),a("p",[t._v("This class is used by the "),a("code",[t._v("Pipeline.to_mlflow()")]),t._v(" method.")]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[t._v("mlflow.pyfunc.model.PythonModel")])]),t._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"load-context"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#load-context"}},[t._v("#")]),t._v(" load_context "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("load_context")]),t._v(" ("),t._v("\n  self,\n  context,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Loads artifacts from the specified :class:"),a("code",[t._v("~PythonModelContext")]),t._v(" that can be used by\n:func:"),a("code",[t._v("~PythonModel.predict")]),t._v(" when evaluating inputs. When loading an MLflow model with\n:func:"),a("code",[t._v("~load_pyfunc")]),t._v(", this method is called as soon as the :class:"),a("code",[t._v("~PythonModel")]),t._v(" is\nconstructed.")]),t._v(" "),a("p",[t._v("The same :class:"),a("code",[t._v("~PythonModelContext")]),t._v(" will also be available during calls to\n:func:"),a("code",[t._v("~PythonModel.predict")]),t._v(", but it may be more efficient to override this method\nand load artifacts from the context at model load time.")]),t._v(" "),a("p",[t._v(":param context: A :class:"),a("code",[t._v("~PythonModelContext")]),t._v(" instance containing artifacts that the model\ncan use to perform inference.")])]),t._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"predict"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#predict"}},[t._v("#")]),t._v(" predict "),a("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("predict")]),t._v(" ("),t._v("\n  self,\n  context,\n  dataframe: pandas.core.frame.DataFrame,\n) \n")]),t._v("\n")])])]),t._v(" "),a("dd",[a("p",[t._v("Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.\nFor more information about the pyfunc input/output API, see the :ref:"),a("code",[t._v("pyfunc-inference-api")]),t._v(".")]),t._v(" "),a("p",[t._v(":param context: A :class:"),a("code",[t._v("~PythonModelContext")]),t._v(" instance containing artifacts that the model\ncan use to perform inference.\n:param model_input: A pyfunc-compatible input for the model to evaluate.")])])])])}),[],!1,null,null,null);e.default=o.exports}}]);