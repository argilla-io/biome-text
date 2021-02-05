# Deployment

*Biome.text* provides an easy to use built-in tool to deploy your model
on a local machine as a REST API via [FastAPI](https://fastapi.tiangolo.com/).
Additionally, you can easily export your pipeline to an
[MLFlow Model](https://mlflow.org/docs/latest/models.html#mlflow-models)
and take advantage of all its deployment tools, like packaging the model as self-contained Docker image
with a REST API endpoint  or deploying it directly on Microsoft Azure ML or Amazon SageMaker.

[[toc]]

## Built-in deployment via FastAPI

The built-in tool uses [FastAPI](https://fastapi.tiangolo.com/) and an [Uvicorn](https://www.uvicorn.org/) server
to expose the model as an API REST service.

### Start the REST service

For the REST service we need to save our pipeline as a `model.tar.gz` file on disk.
This can be achieved by either training our pipeline with `Pipeline.train()`, in which case the `model.tar.gz` file is part
of the training output, or by simply calling `Pipeline.save()` to serialize the pipeline in its current state.
With the `model.tar.gz` file at hand we use the *biome.text* CLI to start the API REST service from the terminal:

```bash
biome serve path/to/output/model.tar.gz
```

If everything is correct, the Uvicorn server starts, and we should see following message in the terminal:

```bash
INFO:     Uvicorn running on http://0.0.0.0:9999 (Press CTRL+C to quit)
```

At this point, everything is up and running.
We can access the documentation of the API in our browsers following this direction: `http://0.0.0.0:9999/docs`

### Quick-tour of the API

The API docs provide an overview of the available endpoints of the REST service:
  - the `/predict` endpoint allows POST requests and is equivalent to the `Pipeline.predict()` method
  - the `/config` endpoint returns the pipeline configuration corresponding to `Pipeline.config`
  - the `/_status` endpoint simply returns the status of the REST service

### Making predictions

The best way to try out the `/predict` endpoint, is through the API docs.
If we open the `/predict` section and click on "*Try it out*", the API will offer us a text field to provide our input.
The text field already provides you with a valid data scheme, and you can simply change the values of the input parameters.
For example, for a pipeline with a `TextClassification` head you could send following request body:

```json
{
	"text": "Hello, test this input",
	"add_tokens": false,
	"add_attributions": true
}
```

If we press **Execute**, we can see the POST call with `Curl` and the request URL to which we sent the request body.
We can also see the server response, with the response code, the response body and the response headers.
The response body should include the prediction corresponding to your input,
or a descriptive error message in case something went wrong.


## Deployment via MLFlow Models

Let us go through a quick example to illustrate how to deploy your *biome.text* models via MLFlow Models.

### Exporting the pipeline to MLFlow

```python{4-9}
from biome.text import Pipeline
import mlflow, pandas

pipeline = Pipeline.from_config({
    "name": "to_mlflow_example",
    "head": {"type": "TextClassification", "labels": ["a", "b"]},
})

model_uri = pipeline.to_mlflow()

model = mlflow.pyfunc.load_model(model_uri)

prediction: pandas.DataFrame = model.predict(pandas.DataFrame([{"text": "Test this text"}]))
```

First we need to export our pipeline as MLFlow model. In this example we use a basic untrained pipeline with a
`TextClassification` head, but normally you would either load a pretrained pipeline via `Pipeline.from_pretrained()`
or train the pipeline first before exporting it.
To export the pipeline, you simply call `.to_mlflow()` that will log your pipeline as MLFlow model on a
MLFlow Tracking Server.
The tracking URI of the server, as well as the run name, and the experiment ID under which to log the model, are
configurable parameters of the method.
The returned string is the artifact URI of the MLFlow model that we can use to load or deploy our model with the
MLFlow deployment tools.

### Loading and deploying the MLFlow Model

```python{11-13}
from biome.text import Pipeline
import mlflow, pandas

pipeline = Pipeline.from_config({
    "name": "to_mlflow_example",
    "head": {"type": "TextClassification", "labels": ["a", "b"]},
})

model_uri = pipeline.to_mlflow()

model = mlflow.pyfunc.load_model(model_uri)

prediction: pandas.DataFrame = model.predict(pandas.DataFrame([{"text": "Test this text"}]))
```

To use the MLFlow model for inference, we feed our model URI to the `mlflow.pyfunc.load_model()` method and call
`.predict()` on the loaded model.
MLFlow models take as input a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
and also return their predictions as a DataFrame.

If we wanted to serve our MLFlow model as a local REST API, we could use the MLFlow CLI command
[`mlflow models`](https://www.mlflow.org/docs/latest/cli.html#mlflow-models):

```bash
mlflow models serve -m <model_uri>
```

:::tip TIP

Do not forget to set the `MLFLOW_TRACKING_URI` environment variable in case you use a
different tracking server location than the default `./mlruns`.

:::

An example request for the served model would be:
```bash
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "columns": ["text"],
    "data": ["test this input", "and this as well"]
}'
```

For more details about how to exploit all MLFlow Model features,
like deploying them on Microsoft Azure ML or Amazon SageMaker, please refer to their
[documentation](https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools).
