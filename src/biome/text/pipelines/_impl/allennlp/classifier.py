import copy
import inspect
import os
from tempfile import mktemp
from typing import Dict, Any, List, Optional, cast, Type, Union

import pandas as pd
import yaml
from allennlp.common import Params
from allennlp.common.from_params import remove_optional
from allennlp.data import Vocabulary, DatasetReader
from allennlp.interpret import SaliencyInterpreter
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.service import server_simple
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from biome.text.defs import (
    PipelineDefinition,
    TextClassifierPipeline as ITextClassifierPipeline,
)
from biome.text.pipelines._impl.allennlp.dataset_readers.configurable_input_datasource_reader import (
    ConfigurableInputDatasourceReader,
)
from biome.text.pipelines._impl.allennlp.interpreters import IntegratedGradient
from biome.text.pipelines._impl.allennlp.learn import learn
from biome.text.pipelines._impl.allennlp.models import load_archive
from biome.text.pipelines._impl.allennlp.models.defs import ITextClassifier
from biome.text.pipelines._impl.allennlp.models.text_classifier import TextClassifier
from biome.text.pipelines._impl.allennlp.predictor import (
    AllenNlpTextClassifierPredictor,
)

DefaultInterpreterClass = IntegratedGradient
from biome.text.pipelines._impl.allennlp import helpers


def _copy_signature(new_signature: inspect.Signature, to_method):
    def wrapper(*args, **kwargs):
        return to_method(*args, **kwargs)

    wrapper.__signature__ = new_signature
    return wrapper


class TextClassifierPipeline(ITextClassifierPipeline):

    """
    Text classifier implementation using the allennlp components and tools.

    The pipeline implementation aim to centralize configuration related to both principal
    components in a AllenNlp model: `allennlp.data.DatasetReader` and `allennlp.models.Model`

    This includes automatic configuration generation, training executions and
    `allennlp.predictors.Predictor` creation.

    This model use as model implementation the `TextClassifier` module, a general text
    classification module that covers a lot of experiments configurations. But if you want create
    your own classifier you can implement the `biome.text.defs.TextClassifierPipeline` from scratch.
    Or, if your implementation is based on allennlp modules, you can extends this class and overwrite
    the base reader and model classes.

    """

    # The inner `allennlp.models.Model` used in this pipeline
    _model_class: ITextClassifier = TextClassifier
    # The inner `allennlp.data.DatasetReader` used in this pipeline
    _dataset_reader_class = ConfigurableInputDatasourceReader

    Model.register(_model_class.__name__, exist_ok=True)(_model_class)
    DatasetReader.register(_dataset_reader_class.__name__, exist_ok=True)(
        _dataset_reader_class
    )

    def __init__(self, name: str):

        super(TextClassifierPipeline, self).__init__(name)

        self._binary = None
        self._config = None
        self._allennlp_config = {}

        self._dataser_reader = None
        self._model = None
        self._predictor = None

        self._labels = []
        self._feature_names = []

    @classmethod
    def from_config(cls, config: PipelineDefinition) -> "TextClassifierPipeline":
        return cls(config.name)._with_config(config)

    @classmethod
    def load(cls, binary_path: str, **kwargs) -> "TextClassifierPipeline":
        # TODO: Find the pipeline name
        return cls("")._with_allennlp_archive(binary_path)

    def learn(
        self,
        trainer: str,
        train: str,
        output: str,
        validation: str = None,
        test: Optional[str] = None,
        vocab: Optional[str] = None,
        verbose: bool = False,
    ) -> "TextClassifierPipeline":
        """
        Launch a learning process for loaded model configuration.

        Once the learn process finish, the model is ready for make predictions

        Parameters
        ----------

        trainer
            The trainer file path
        train
            The train datasource file path
        validation
            The validation datasource file path
        output
            The learn output path
        vocab: Vocab
            The already generated vocabulary path
        test: str
            The test datasource configuration
        verbose
            Turn on verbose logs
        """

        kwargs = dict(
            vocab=vocab,
            test_cfg=test,
            output=output,
            trainer_path=trainer,
            train_cfg=train,
            validation_cfg=validation,
            verbose=verbose,
        )

        if self._binary:
            learn(model_binary=self._binary, **kwargs)
        else:
            spec = mktemp()
            with open(spec, "wt") as file:
                yaml.safe_dump(self.allennlp_config(), file)
            _ = learn(model_spec=spec, **kwargs)

        return self._with_allennlp_archive(os.path.join(output, "model.tar.gz"))

    def serve(self, port: int, predictions: str):
        if predictions:
            self.init_prediction_logger(predictions)

        app = server_simple.make_app(self._predictor, title=self.name)
        CORS(app)

        http_server = WSGIServer(("0.0.0.0", port), app)
        self._LOGGER.info("Model loaded, serving on port %s", port)

        http_server.serve_forever()

    def _interpret_dataframe(
        self, df: pd.DataFrame, interpreter_klass: Type = DefaultInterpreterClass
    ) -> pd.Series:
        def interpret_row(
            row: pd.Series, interpreter: SaliencyInterpreter
        ) -> Union[dict, List[dict]]:
            """Interpret a incoming dataframe row"""
            data = row.to_dict()
            interpretation = interpreter.saliency_interpret_from_json(cast(dict, data))
            if len(interpretation) == 0:
                return {}
            if len(interpretation) == 1:
                return interpretation[0]
            return interpretation

        interpreter = interpreter_klass(self._predictor)
        return df.apply(interpret_row, interpreter=interpreter, axis=1)

    def get_output_labels(self) -> List[str]:
        return cast(
            AllenNlpTextClassifierPredictor, self._predictor
        ).get_output_labels()

    def allennlp_config(self) -> Dict[str, Any]:
        return self._allennlp_config.copy()

    def predict(self, *inputs, **kw_inputs) -> Dict[str, Any]:
        if not self._predictor:
            return {}

        kw_inputs.update(
            {
                self.inputs_keys()[idx]: input_value
                for idx, input_value in enumerate(inputs)
            }
        )

        output = self._predictor.predict_json(kw_inputs)
        self.log_prediction(kw_inputs, output)

        return output

    def inputs_keys(self) -> List[str]:
        return self._config.inputs

    def output(self) -> str:
        return self._config.output

    def _empty_predictor(self, inputs: List[str]) -> AllenNlpTextClassifierPredictor:
        """Creates a dummy pipeline with labels for model layers"""
        vocab = self._empty_vocab()

        return AllenNlpTextClassifierPredictor(
            model=Model.from_params(
                Params(copy.deepcopy(self._allennlp_config["model"])), vocab=vocab
            ),
            dataset_reader=DatasetReader.from_params(
                Params(copy.deepcopy(self._allennlp_config["dataset_reader"]))
            ),
        )

    def _empty_vocab(self, labels: List[str] = None) -> Vocabulary:
        """
        This method generate a mock vocabulary for the 3 common allennlp namespaces.
        If default model use another tokens indexer key name, the pipeline model won't be loaded
        from configuration
        """
        labels = labels or ["true", "false"]
        vocab = Vocabulary()

        vocab.add_tokens_to_namespace(labels, namespace="labels")
        for namespace in self._feature_names:
            vocab.add_token_to_namespace("t", namespace=namespace)

        return vocab

    def extend_labels(self, labels: List[str]) -> None:
        if self._predictor:
            self._predictor.extend_labels(labels)

    def _with_config(self, config: PipelineDefinition) -> "TextClassifierPipeline":

        self._config = config
        self._feature_names = [feature for feature in config.textual_features.keys()]
        self._allennlp_config = self._pipeline_definition_to_allennlp_config(config)
        self._predictor = self._empty_predictor(inputs=self._config.inputs)
        self._update_predict_signature()

        return self

    def _update_predict_signature(self):
        setattr(
            self,
            "predict",
            _copy_signature(
                inspect.Signature(
                    [
                        inspect.Parameter(input_name, inspect.Parameter.KEYWORD_ONLY)
                        for input_name in self.inputs_keys()
                    ]
                ),
                self.predict,
            ),
        )

    def _with_allennlp_archive(self, path: str) -> "TextClassifierPipeline":
        archive = load_archive(path)

        self._allennlp_config = {
            "dataset_reader": cast(Params, archive.config["dataset_reader"]).as_dict(),
            "model": cast(Params, archive.config["model"]).as_dict(),
        }

        model = archive.model
        reader = DatasetReader.from_params(
            Params(self._allennlp_config["dataset_reader"].copy())
        )

        if not isinstance(model, self._model_class):
            raise TypeError(
                f"Model type for model {model} not supported. Expected: {self._model_class}"
            )

        if not isinstance(reader, self._dataset_reader_class):
            raise TypeError(
                f"Reader type for reader {reader} not supported. Expected: {self._dataset_reader_class}"
            )

        self._predictor = AllenNlpTextClassifierPredictor(
            model=model, dataset_reader=reader
        )
        self._config = self._allennlp_config_to_pipeline_definition(
            self._allennlp_config
        )
        self._binary = path
        self._update_predict_signature()

        return self

    def _pipeline_definition_to_allennlp_config(
        self, config: PipelineDefinition
    ) -> Dict[str, Any]:
        text_field_embedder = {
            feature: config["embedder"]
            for feature, config in config.textual_features.items()
        }
        vocab = self._empty_vocab()
        return {
            "dataset_reader": self._dataset_reader_from_definition(config),
            "model": self._model_from_definition(config, text_field_embedder, vocab),
        }

    def _model_from_definition(
        self,
        config: PipelineDefinition,
        embedder_config: Dict[str, Any],
        vocab: Vocabulary,
    ) -> Dict[str, Any]:

        init_model_signature = inspect.signature(self._model_class.__init__)
        embedder_name = None
        for param in init_model_signature.parameters.values():
            if param.annotation and issubclass(param.annotation, TextFieldEmbedder):
                embedder_name = param.name
                break

        return {
            "type": self._model_class.__name__,
            embedder_name: embedder_config,
            **config.architecture,
        }

    def _dataset_reader_from_definition(
        self, config: PipelineDefinition
    ) -> Dict[str, Any]:
        token_indexers = {
            feature: config["indexer"]
            for feature, config in config.textual_features.items()
        }

        return {
            "type": self._dataset_reader_class.__name__,
            "tokenizer": config.tokenizer,
            "token_indexers": token_indexers,
            "inputs": self._config.inputs,
            "output": self._config.output,
            "aggregate_inputs": self._config.aggregate_inputs,
        }

    def _allennlp_config_to_pipeline_definition(
        self, allennlp_config: Dict[str, Any]
    ) -> PipelineDefinition:

        dataset_reader = copy.deepcopy(allennlp_config["dataset_reader"])
        model = copy.deepcopy(allennlp_config["model"])
        model.pop("type")

        tokenizer = dataset_reader["tokenizer"]
        inputs = dataset_reader["inputs"]
        output = dataset_reader.get("output")

        token_indexers = dataset_reader.pop("token_indexers")
        text_field_embedder = model.get("text_field_embedder", model.get("embedder"))

        textual_features = {
            feature: {"indexer": v, "embedder": text_field_embedder[feature]}
            for feature, v in token_indexers.items()
        }

        return PipelineDefinition(
            type=self._model_class.__name__,
            tokenizer=tokenizer,
            inputs=inputs,
            output=output,
            textual_features=textual_features,
            architecture={layer: config for layer, config in model.items()},
        )
