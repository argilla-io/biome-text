# Deployment

*Biome.text* provides an easy to use built-in tool to deploy your model on a local machine as a REST API via [FastAPI](https://fastapi.tiangolo.com/).
Additionally, you can easily export your pipeline to an [MLFlow Model](https://mlflow.org/docs/latest/models.html#mlflow-models)
and take advantage of all its deployment tools like packaging the model as self-contained Docker image with a REST API endpoint,
or deploying it directly on Microsoft Azure ML or Amazon SageMaker.

## Built-in deployment via FastAPI
### Start the REST endpoint

Once we have defined and trained a model, and endpoint with the API REST can be established using the command

```bash
biome serve path/to/output/model.tar.gz
```

If everything is correct, the server process will begin by the application startup, and when it's done, we will receive the following message in the terminal:

```bash
INFO:     Uvicorn running on http://0.0.0.0:9999 (Press CTRL+C to quit)
```

At this point, everything is up and running. We can access the documentation of the API in our browsers following this direction: `http://0.0.0.0:9999/docs`

### Quick-tour of the API

When the API docs are open, the defaults calls that we can make are shown to us. These calls are the ones provided by [FastAPI](https://fastapi.tiangolo.com/), like *config*, *status* and *predict*

### Making predictions

As we came from a trained model, we can make predictions right away. We can use the predict POST call, which is equivalent to the `Pipeline.predict()` made on Python.

If we open the predict section and click on "*Try it out*", the API will offer us a text field to write out our message. The input text must be structured as a Python dictionary, so we can write:

```python
{
	"text": "Hello",
	"add_tokens": false,
	"add_attributions": false
}
```

If we press "*Execute*", we can see the POST call in Curl and the request URL at which we send the call. We can also see the server response, with the response call, the response body and headers, and a description of the call in case it can be useful (e.g. in an error response).

## Deployment via MLFlow Models
