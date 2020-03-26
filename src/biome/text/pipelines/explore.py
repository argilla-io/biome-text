import logging

from dask import dataframe as dd

from biome.text import Pipeline
from biome.text.pipelines.defs import ExploreConfig, ElasticsearchConfig

__LOGGER = logging.getLogger(__name__)


def pipeline_predictions(
    pipeline: Pipeline,
    source_path: str,
    config: ExploreConfig,
    es_config: ElasticsearchConfig,
) -> dd.DataFrame:
    """
    Read a data source and tries to apply a model predictions to the whole data source. The
    results will be persisted into an elasticsearch index for further data exploration

    """
    return pipeline.explore(source_path, config, es_config)


def register_biome_prediction(
    name: str, pipeline: Pipeline, es_config: ElasticsearchConfig, **kwargs
):
    pipeline.register_biome_prediction(name, es_config, **kwargs)
