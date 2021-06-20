(window.webpackJsonp=window.webpackJsonp||[]).push([[63],{472:function(t,e,a){"use strict";a.r(e);var s=a(27),n=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"deployment"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#deployment"}},[t._v("#")]),t._v(" Deployment")]),t._v(" "),a("p",[a("em",[t._v("Biome.text")]),t._v(" provides an easy to use built-in tool to deploy your model\non a local machine as a REST API via "),a("a",{attrs:{href:"https://fastapi.tiangolo.com/",target:"_blank",rel:"noopener noreferrer"}},[t._v("FastAPI"),a("OutboundLink")],1),t._v(".\nAdditionally, you can easily export your pipeline to an\n"),a("a",{attrs:{href:"https://mlflow.org/docs/latest/models.html#mlflow-models",target:"_blank",rel:"noopener noreferrer"}},[t._v("MLFlow Model"),a("OutboundLink")],1),t._v("\nand take advantage of all its deployment tools, like packaging the model as self-contained Docker image\nwith a REST API endpoint  or deploying it directly on Microsoft Azure ML or Amazon SageMaker.")]),t._v(" "),a("p"),a("div",{staticClass:"table-of-contents"},[a("ul",[a("li",[a("a",{attrs:{href:"#built-in-deployment-via-fastapi"}},[t._v("Built-in deployment via FastAPI")]),a("ul",[a("li",[a("a",{attrs:{href:"#start-the-rest-service"}},[t._v("Start the REST service")])]),a("li",[a("a",{attrs:{href:"#quick-tour-of-the-api"}},[t._v("Quick-tour of the API")])]),a("li",[a("a",{attrs:{href:"#making-predictions"}},[t._v("Making predictions")])])])]),a("li",[a("a",{attrs:{href:"#deployment-via-mlflow-models"}},[t._v("Deployment via MLFlow Models")]),a("ul",[a("li",[a("a",{attrs:{href:"#exporting-the-pipeline-to-mlflow"}},[t._v("Exporting the pipeline to MLFlow")])]),a("li",[a("a",{attrs:{href:"#loading-and-deploying-the-mlflow-model"}},[t._v("Loading and deploying the MLFlow Model")])])])])])]),a("p"),t._v(" "),a("h2",{attrs:{id:"built-in-deployment-via-fastapi"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#built-in-deployment-via-fastapi"}},[t._v("#")]),t._v(" Built-in deployment via FastAPI")]),t._v(" "),a("p",[t._v("The built-in tool uses "),a("a",{attrs:{href:"https://fastapi.tiangolo.com/",target:"_blank",rel:"noopener noreferrer"}},[t._v("FastAPI"),a("OutboundLink")],1),t._v(" and an "),a("a",{attrs:{href:"https://www.uvicorn.org/",target:"_blank",rel:"noopener noreferrer"}},[t._v("Uvicorn"),a("OutboundLink")],1),t._v(" server\nto expose the model as an API REST service.")]),t._v(" "),a("h3",{attrs:{id:"start-the-rest-service"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#start-the-rest-service"}},[t._v("#")]),t._v(" Start the REST service")]),t._v(" "),a("p",[t._v("For the REST service we need to save our pipeline as a "),a("code",[t._v("model.tar.gz")]),t._v(" file on disk.\nThis can be achieved by either training our pipeline with "),a("code",[t._v("Pipeline.train()")]),t._v(", in which case the "),a("code",[t._v("model.tar.gz")]),t._v(" file is part\nof the training output, or by simply calling "),a("code",[t._v("Pipeline.save()")]),t._v(" to serialize the pipeline in its current state.\nWith the "),a("code",[t._v("model.tar.gz")]),t._v(" file at hand we use the "),a("em",[t._v("biome.text")]),t._v(" CLI to start the API REST service from the terminal:")]),t._v(" "),a("div",{staticClass:"language-bash extra-class"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[t._v("biome serve path/to/output/model.tar.gz\n")])])]),a("p",[t._v("If everything is correct, the Uvicorn server starts, and we should see following message in the terminal:")]),t._v(" "),a("div",{staticClass:"language-bash extra-class"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[t._v("INFO:     Uvicorn running on http://0.0.0.0:9999 "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("Press CTRL+C to quit"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),a("p",[t._v("At this point, everything is up and running.\nWe can access the documentation of the API in our browsers following this direction: "),a("code",[t._v("http://0.0.0.0:9999/docs")])]),t._v(" "),a("h3",{attrs:{id:"quick-tour-of-the-api"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#quick-tour-of-the-api"}},[t._v("#")]),t._v(" Quick-tour of the API")]),t._v(" "),a("p",[t._v("The API docs provide an overview of the available endpoints of the REST service:")]),t._v(" "),a("ul",[a("li",[t._v("the "),a("code",[t._v("/predict")]),t._v(" endpoint allows POST requests and is equivalent to the "),a("code",[t._v("Pipeline.predict()")]),t._v(" method")]),t._v(" "),a("li",[t._v("the "),a("code",[t._v("/config")]),t._v(" endpoint returns the pipeline configuration corresponding to "),a("code",[t._v("Pipeline.config")])]),t._v(" "),a("li",[t._v("the "),a("code",[t._v("/_status")]),t._v(" endpoint simply returns the status of the REST service")])]),t._v(" "),a("h3",{attrs:{id:"making-predictions"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#making-predictions"}},[t._v("#")]),t._v(" Making predictions")]),t._v(" "),a("p",[t._v("The best way to try out the "),a("code",[t._v("/predict")]),t._v(" endpoint, is through the API docs.\nIf we open the "),a("code",[t._v("/predict")]),t._v(' section and click on "'),a("em",[t._v("Try it out")]),t._v('", the API will offer us a text field to provide our input.\nThe text field already provides you with a valid data scheme, and you can simply change the values of the input parameters.\nFor example, for a pipeline with a '),a("code",[t._v("TextClassification")]),t._v(" head you could send following request body:")]),t._v(" "),a("div",{staticClass:"language-json extra-class"},[a("pre",{pre:!0,attrs:{class:"language-json"}},[a("code",[a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n\t"),a("span",{pre:!0,attrs:{class:"token property"}},[t._v('"text"')]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Hello, test this input"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n\t"),a("span",{pre:!0,attrs:{class:"token property"}},[t._v('"add_tokens"')]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("false")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n\t"),a("span",{pre:!0,attrs:{class:"token property"}},[t._v('"add_attributions"')]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("true")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n")])])]),a("p",[t._v("If we press "),a("strong",[t._v("Execute")]),t._v(", we can see the POST call with "),a("code",[t._v("Curl")]),t._v(" and the request URL to which we sent the request body.\nWe can also see the server response, with the response code, the response body and the response headers.\nThe response body should include the prediction corresponding to your input,\nor a descriptive error message in case something went wrong.")]),t._v(" "),a("h2",{attrs:{id:"deployment-via-mlflow-models"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#deployment-via-mlflow-models"}},[t._v("#")]),t._v(" Deployment via MLFlow Models")]),t._v(" "),a("p",[t._v("Let us go through a quick example to illustrate how to deploy your "),a("em",[t._v("biome.text")]),t._v(" models via MLFlow Models.")]),t._v(" "),a("h3",{attrs:{id:"exporting-the-pipeline-to-mlflow"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#exporting-the-pipeline-to-mlflow"}},[t._v("#")]),t._v(" Exporting the pipeline to MLFlow")]),t._v(" "),a("div",{staticClass:"language-python extra-class"},[a("div",{staticClass:"highlight-lines"},[a("br"),a("br"),a("br"),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("br"),a("br"),a("br"),a("br"),a("br")]),a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("from")]),t._v(" biome"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("text "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" Pipeline\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" mlflow"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" pandas\n\npipeline "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Pipeline"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("from_config"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"name"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"to_mlflow_example"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"head"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"type"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"TextClassification"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"labels"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"a"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"b"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nmodel_uri "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" pipeline"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("to_mlflow"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nmodel "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" mlflow"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("pyfunc"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("load_model"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("model_uri"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nprediction"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" pandas"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("DataFrame "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" model"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("predict"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("pandas"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("DataFrame"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"text"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Test this text"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),a("p",[t._v("First we need to export our pipeline as MLFlow model. In this example we use a basic untrained pipeline with a\n"),a("code",[t._v("TextClassification")]),t._v(" head, but normally you would either load a pretrained pipeline via "),a("code",[t._v("Pipeline.from_pretrained()")]),t._v("\nor train the pipeline first before exporting it.\nTo export the pipeline, you simply call "),a("code",[t._v(".to_mlflow()")]),t._v(" that will log your pipeline as MLFlow model on a\nMLFlow Tracking Server.\nThe tracking URI of the server, as well as the run name, and the experiment ID under which to log the model, are\nconfigurable parameters of the method.\nThe returned string is the artifact URI of the MLFlow model that we can use to load or deploy our model with the\nMLFlow deployment tools.")]),t._v(" "),a("h3",{attrs:{id:"loading-and-deploying-the-mlflow-model"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#loading-and-deploying-the-mlflow-model"}},[t._v("#")]),t._v(" Loading and deploying the MLFlow Model")]),t._v(" "),a("div",{staticClass:"language-python extra-class"},[a("div",{staticClass:"highlight-lines"},[a("br"),a("br"),a("br"),a("br"),a("br"),a("br"),a("br"),a("br"),a("br"),a("br"),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("div",{staticClass:"highlighted"},[t._v(" ")]),a("br")]),a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("from")]),t._v(" biome"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("text "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" Pipeline\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" mlflow"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" pandas\n\npipeline "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Pipeline"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("from_config"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"name"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"to_mlflow_example"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"head"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"type"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"TextClassification"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"labels"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"a"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"b"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nmodel_uri "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" pipeline"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("to_mlflow"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nmodel "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" mlflow"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("pyfunc"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("load_model"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("model_uri"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\nprediction"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" pandas"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("DataFrame "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" model"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("predict"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("pandas"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("DataFrame"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"text"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Test this text"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),a("p",[t._v("To use the MLFlow model for inference, we feed our model URI to the "),a("code",[t._v("mlflow.pyfunc.load_model()")]),t._v(" method and call\n"),a("code",[t._v(".predict()")]),t._v(" on the loaded model.\nMLFlow models take as input a "),a("a",{attrs:{href:"https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html",target:"_blank",rel:"noopener noreferrer"}},[t._v("pandas DataFrame"),a("OutboundLink")],1),t._v("\nand also return their predictions as a DataFrame.")]),t._v(" "),a("p",[t._v("If we wanted to serve our MLFlow model as a local REST API, we could use the MLFlow CLI command\n"),a("a",{attrs:{href:"https://www.mlflow.org/docs/latest/cli.html#mlflow-models",target:"_blank",rel:"noopener noreferrer"}},[a("code",[t._v("mlflow models")]),a("OutboundLink")],1),t._v(":")]),t._v(" "),a("div",{staticClass:"language-bash extra-class"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[t._v("mlflow models serve -m "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v("model_uri"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v("\n")])])]),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("TIP")]),t._v(" "),a("p",[t._v("Do not forget to set the "),a("code",[t._v("MLFLOW_TRACKING_URI")]),t._v(" environment variable in case you use a\ndifferent tracking server location than the default "),a("code",[t._v("./mlruns")]),t._v(".")])]),t._v(" "),a("p",[t._v("An example request for the served model would be:")]),t._v(" "),a("div",{staticClass:"language-bash extra-class"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("curl")]),t._v(" http://127.0.0.1:5000/invocations -H "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'Content-Type: application/json'")]),t._v(" -d "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('\'{\n    "columns": ["text"],\n    "data": ["test this input", "and this as well"]\n}\'')]),t._v("\n")])])]),a("p",[t._v("For more details about how to exploit all MLFlow Model features,\nlike deploying them on Microsoft Azure ML or Amazon SageMaker, please refer to their\n"),a("a",{attrs:{href:"https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools",target:"_blank",rel:"noopener noreferrer"}},[t._v("documentation"),a("OutboundLink")],1),t._v(".")])])}),[],!1,null,null,null);e.default=n.exports}}]);