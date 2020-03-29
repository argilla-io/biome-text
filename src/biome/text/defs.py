import datetime
import json
import logging
import os
import pickle
import warnings
from abc import ABC
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, List, Type

import numpy
import pandas as pd
import yaml
from allennlp.common.util import sanitize
from biome.data import DataSource
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient

import biome.text.helpers
from biome.text import constants
from biome.text.pipelines.defs import ExploreConfig, ElasticsearchConfig


class _HashDict(dict):
    """
    hashable dict implementation.
    BE AWARE! Since dicts are mutable, the hash can change!
    """

    def __hash__(self):
        # user a better way
        return pickle.dumps(self).__hash__()


class _HashList(list):
    def __hash__(self):
        return pickle.dumps(self).__hash__()


_TYPES_MAP: Dict[str, Type["Pipeline"]] = {}


class Pipeline:
    """Base pipeline interface"""

    PREDICTION_FILE_NAME = "predictions.json"

    _LOGGER = logging.getLogger(__name__)

    def __init__(self, name: str):
        self._name = name
        self._prediction_logger = None

    @classmethod
    def register(cls, pipeline_type: str, overrides: bool = False):
        """Register a new pipeline class for a given type name"""
        if overrides or pipeline_type not in _TYPES_MAP:
            _TYPES_MAP[pipeline_type] = cls

    @classmethod
    def by_type(cls, pipeline_type: str) -> Optional[Type["Pipeline"]]:
        """Get an already registered pipeline class for a given type name, if exists"""
        return _TYPES_MAP.get(pipeline_type)

    @property
    def name(self) -> str:
        """The pipeline name"""
        return self._name

    @classmethod
    def from_config(cls, config: "PipelineDefinition") -> "Pipeline":
        """Creates a pipeline from """
        raise NotImplementedError

    @classmethod
    def load(cls, binary: str, **kwargs) -> "Pipeline":
        raise NotImplementedError

    def predict(self, *inputs, **kw_inputs) -> Dict[str, Any]:
        """Pipeline prediction/inference method"""
        raise NotImplementedError

    def learn(
        self,
        output: str,
        trainer: str,
        train: str,
        validation: Optional[str] = None,
        test: Optional[str] = None,
        # This parameter is hard coupled to allennlp implementation details
        # Could (not) be sense define in a common way an optional vocab for learn phase
        vocab: Optional[str] = None,
        verbose: bool = False,
    ) -> "Pipeline":
        """Learn method for create a trained pipeline"""
        raise NotImplementedError

    def init_prediction_cache(self, max_size: int) -> None:
        """Initialize a prediction cache using the functools.lru_cache decorator

        Parameters
        ----------
        max_size
            Save up to max_size most recent items.
        """

        if hasattr(self, "_predict_with_cache"):
            warnings.warn(
                "Prediction cache already initiated!", category=RuntimeWarning
            )
            return

        predict_with_cache = lru_cache(maxsize=max_size)(self.predict)

        def predict_wrapper(*args, **kwargs):
            def hashable_value(value) -> Any:
                if isinstance(value, dict):
                    return _HashDict(value)
                if isinstance(value, (list, tuple)):
                    return _HashList(value)
                return value

            return predict_with_cache(
                *[hashable_value(arg_value) for arg_value in args],
                **{
                    key: hashable_value(input_value)
                    for key, input_value in kwargs.items()
                },
            )

        self.__setattr__("predict", predict_wrapper)
        self.__setattr__("_predict_with_cache", predict_with_cache)

    def init_prediction_logger(
        self, output_dir: str, max_bytes: int = 20000000, backup_count: int = 20
    ):
        """Initialize the prediction logger.

        If initialized we will log all predictions to a file called *predictions.json* in the `output_folder`.

        Parameters
        ----------
        output_dir
            Path to the folder in which we create the *predictions.json* file.
        max_bytes
            Passed on to logging.handlers.RotatingFileHandler
        backup_count
            Passed on to logging.handlers.RotatingFileHandler

        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        predictions_logger = logging.getLogger(output_dir)
        predictions_logger.setLevel(logging.DEBUG)
        # This flag avoids logging messages to be propagated to the parent loggers
        predictions_logger.propagate = False
        file_handler = RotatingFileHandler(
            os.path.join(output_dir, self.PREDICTION_FILE_NAME),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(logging.INFO)
        predictions_logger.addHandler(file_handler)

        self._prediction_logger = predictions_logger

    def log_prediction(
        self, inputs: Dict[str, Any], prediction: Dict[str, Any]
    ) -> None:
        if self._prediction_logger:
            self._prediction_logger.info(
                json.dumps(dict(inputs=inputs, annotation=prediction))
            )

    def explore(
        self, ds_path: str, config: ExploreConfig, es_config: ElasticsearchConfig,
    ):
        """
            Read a data source and tries to apply a model predictions to the whole data source. The
            results will be persisted into an elasticsearch index for further data exploration
        """

        if config.prediction_cache > 0:
            self.init_prediction_cache(config.prediction_cache)

        data_source = DataSource.from_yaml(ds_path)
        ddf_mapped = data_source.to_mapped_dataframe()
        # this only makes really sense when we have a predict_batch_json method implemented ...
        n_partitions = max(1, round(len(ddf_mapped) / config.batch_size))

        # a persist is necessary here, otherwise it fails for n_partitions == 1
        # the reason is that with only 1 partition we pass on a generator to predict_batch_json
        ddf_mapped = ddf_mapped.repartition(npartitions=n_partitions).persist()
        ddf_mapped_columns = ddf_mapped.columns

        ddf_mapped["annotation"] = ddf_mapped[self.inputs_keys()].apply(
            lambda x: sanitize(self.predict(**x.to_dict())),
            axis=1,
            meta=(None, object),
        )

        if config.interpret:
            # TODO we should apply the same mechanism for the model predictions. Creating a new pipeline
            #  for every partition
            ddf_mapped["interpretations"] = ddf_mapped[
                ddf_mapped_columns
            ].map_partitions(self._interpret_dataframe, meta=(None, object))

        ddf_source = data_source.to_dataframe()
        ddf_source = ddf_source.repartition(npartitions=n_partitions).persist()

        # We are sure that both data frames are aligned!
        # A 100% safe way would be to set_index of both data frames on a meaningful column.
        # The main problem are multiple csv files (read_csv("*.csv")), where the index starts from 0 for each file ...
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ddf = dd.concat([ddf_source, ddf_mapped], axis=1)

        # TODO @dcfidalgo we could calculate base metrics here (F1, recall & precision) using dataframe.
        #  And include as part of explore metadata
        #  Does it's simple???

        ddf = DaskElasticClient(
            host=es_config.es_host, retry_on_timeout=True, http_compress=True
        ).save(ddf, index=es_config.es_index, doc_type=es_config.es_doc)

        merged_metadata = {
            **(config.metadata or {}),
            "datasource": ds_path,
            # TODO this should change when ui is normalized (action detail and action link naming)F
            "explore_name": es_config.es_index,
            "model": self.name,
            "columns": ddf.columns.values.tolist(),
        }

        self._register_biome_prediction(
            name=es_config.es_index, es_config=es_config, **merged_metadata
        )
        self._prepare_es_index(es_config, force_delete=config.force_delete)
        ddf = ddf.persist()
        self._LOGGER.info(
            "Data annotated successfully. You can explore your data here: %s",
            f"{constants.EXPLORE_APP_ENDPOINT}/projects/default/explore/{es_config.es_index}",
        )

        return ddf

    def _interpret_dataframe(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Apply a model interpretation to every partition dataframe

        Parameters
        ----------
        df: pd.DataFrame
            The partition DataFrame

        Returns
        -------

        A pandas Series representing the interpretations

        """
        raise NotImplementedError

    def serve(self, port: int, predictions: str):
        # TODO: we can generalize the serve action
        raise NotImplementedError

    def inputs_keys(self) -> List[str]:
        raise NotImplementedError

    def output(self) -> str:
        raise NotImplementedError

    def _register_biome_prediction(
        self, name: str, es_config: ElasticsearchConfig, **extra_metadata
    ):
        """
        Creates a new metadata entry for the incoming prediction

        Parameters
        ----------
        name
            A descriptive prediction name
        pipeline
            The pipeline used for the prediction batch
        es_config:
            The Elasticsearch configuration data
        extra_metadata
            Extra arguments passed as extra metadata info
        """

        metadata_index = constants.BIOME_METADATA_INDEX

        es_config.client.indices.create(
            index=metadata_index,
            body={
                "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}
            },
            params=dict(ignore=400),
        )

        parameters = {
            **extra_metadata,
            "pipeline": self.name,
            "signature": self.inputs_keys() + [self.output()],
            "predict_signature": self.inputs_keys(),
            # TODO remove when ui is adapted
            "inputs": self.inputs_keys(),  # backward compatibility
        }

        es_config.client.update(
            index=metadata_index,
            doc_type=es_config.es_doc,
            id=es_config.es_index,
            body={
                "doc": dict(
                    name=name, created_at=datetime.datetime.now(), **parameters
                ),
                "doc_as_upsert": True,
            },
        )

    @staticmethod
    def _prepare_es_index(es_config: ElasticsearchConfig, force_delete: bool):
        dynamic_templates = [
            {
                data_type: {
                    "match_mapping_type": data_type,
                    "path_match": path_match,
                    "mapping": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                }
            }
            for data_type, path_match in [("*", "*.value"), ("string", "*")]
        ]

        if force_delete:
            es_config.client.indices.delete(index=es_config.es_index, ignore=[400, 404])

        es_config.client.indices.create(
            index=es_config.es_index,
            body={
                "mappings": {es_config.es_doc: {"dynamic_templates": dynamic_templates}}
            },
            ignore=400,
        )


class TextClassifierPipeline(Pipeline, ABC):
    """
    A common classifier pipeline class definition
    """

    def extend_labels(self, labels: List[str]) -> "TextClassifierPipeline":
        raise NotImplementedError

    def get_output_labels(self) -> List[str]:
        raise NotImplementedError


class PipelineDefinition:
    """A common pipeline builder"""

    def __init__(
        self,
        type: str,
        inputs: List[str],
        tokenizer: Dict[str, Any],
        textual_features: Dict[str, Dict[str, Any]],
        aggregate_inputs: bool = True,
        multilabel: Optional[bool] = False,
        name: Optional[str] = None,
        output: Optional[str] = None,
        **architecture: Dict[str, Dict[str, Any]],
    ):
        self.name = name or "noname"
        self.inputs = inputs
        self.output = output or "label"
        self.aggregate_inputs = aggregate_inputs
        self.type = type
        self.tokenizer = tokenizer
        self.textual_features = textual_features
        self.architecture = architecture
        self.multilabel = multilabel
        self.inference = {}

    @staticmethod
    def from_config(config: str) -> "PipelineDefinition":
        data = yaml.safe_load(config)
        return PipelineDefinition(**data)

    @staticmethod
    def from_file(path: str) -> "PipelineDefinition":
        data = biome.text.helpers.yaml_to_dict(path)
        return PipelineDefinition(**data)

    def prediction_cache(self, max_size: int) -> "PipelineDefinition":
        self.inference["cache.max_size"] = max_size
        return self

    def prediction_storage(
        self, output_dir: str, max_bytes: int = 20000000, backup_count: int = 20
    ) -> "PipelineDefinition":
        self.inference["predictions.logger"] = {
            "output_dir": output_dir,
            "max_bytes": max_bytes,
            "backup_count": backup_count,
        }
        return self
