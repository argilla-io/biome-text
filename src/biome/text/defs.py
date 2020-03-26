import logging
import os
from abc import ABC
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, List, Type

import yaml
from biome.data import DataSource

import pandas as pd

import biome.text.helpers
from biome.text.pipelines.defs import ExploreConfig, ElasticsearchConfig


class Pipeline:
    """Base pipeline interface"""

    def __init__(self, name: str):
        self._name = name

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

    def explore(
        self,
        data_source: DataSource,
        config: ExploreConfig,
        es_config: ElasticsearchConfig,
    ):
        """
            Read a data source and tries to apply a model predictions to the whole data source. The
            results will be persisted into an elasticsearch index for further data exploration
        """
        raise NotImplementedError

    def serve(self, port: int, predictions: str):
        raise NotImplementedError

    def inputs_keys(self) -> List[str]:
        raise NotImplementedError

    def output(self) -> str:
        raise NotImplementedError

    def init_prediction_logger(self, output_dir: str, **kwargs):
        raise NotImplementedError


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
        architecture: Dict[str, Dict[str, Any]],
        name: Optional[str] = None,
        output: Optional[str] = None,
    ):
        self.name = name or "noname"
        self.inputs = inputs
        self.output = output or "label"
        self.type = type
        self.tokenizer = tokenizer
        self.textual_features = textual_features
        self.architecture = architecture
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
