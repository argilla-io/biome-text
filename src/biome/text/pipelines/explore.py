import datetime
import logging
import warnings
from typing import Type, Union, List

import pandas as pd
from allennlp.interpret import SaliencyInterpreter
from dask import dataframe as dd
from dask_elk.client import DaskElasticClient

from biome.data import DataSource
from biome.text import Pipeline
from biome.text import constants
from biome.text.interpreters import IntegratedGradient as DefaultInterpreterClass
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

    if config.prediction_cache > 0:
        pipeline.init_prediction_cache(config.prediction_cache)

    ds = DataSource.from_yaml(source_path)
    ddf_mapped = ds.to_mapped_dataframe()
    # this only makes really sense when we have a predict_batch_json method implemented ...
    n_partitions = max(1, round(len(ddf_mapped) / config.batch_size))

    # a persist is necessary here, otherwise it fails for n_partitions == 1
    # the reason is that with only 1 partition we pass on a generator to predict_batch_json
    ddf_mapped = ddf_mapped.repartition(npartitions=n_partitions).persist()
    ddf_mapped_columns = ddf_mapped.columns

    ddf_mapped["annotation"] = ddf_mapped[ddf_mapped_columns].apply(
        lambda x: pipeline.predict_json(x.to_dict()), axis=1, meta=(None, object)
    )

    if config.interpret:
        # TODO we should apply the same mechanism for the model predictions. Creating a new pipeline
        #  for every partition
        ddf_mapped["interpretations"] = ddf_mapped[ddf_mapped_columns].map_partitions(
            _interpret_dataframe, pipeline=pipeline, meta=(None, object)
        )

    ddf_source = ds.to_dataframe()
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
        **dict(
            datasource=source_path,
            # TODO this should change when ui is normalized (action detail and action link naming)F
            explore_name=es_config.es_index,
            model=pipeline.name,
            columns=ddf.columns.values.tolist(),
        ),
        **(config.metadata or {}),
    }

    register_biome_prediction(
        name=es_config.es_index,
        pipeline=pipeline,
        es_config=es_config,
        **merged_metadata,
    )

    __prepare_es_index(es_config, force_delete=config.force_delete)
    ddf = ddf.persist()
    __LOGGER.info(
        "Data annotated successfully. You can explore your data here: %s",
        f"{constants.EXPLORE_APP_ENDPOINT}/projects/default/explore/{es_config.es_index}",
    )
    return ddf


def register_biome_prediction(
    name: str, pipeline: Pipeline, es_config: ElasticsearchConfig, **kwargs
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
    kwargs
        Extra arguments passed as extra metadata info
    """

    metadata_index = constants.BIOME_METADATA_INDEX

    es_config.client.indices.create(
        index=metadata_index,
        body={"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}},
        params=dict(ignore=400),
    )

    predict_signature = [
        k for k, v in pipeline.signature.items() if not v.get("optional")
    ]
    parameters = {
        **kwargs,
        "pipeline": pipeline.name,
        "signature": list(pipeline.signature.keys()),
        "predict_signature": predict_signature,
        # TODO remove when ui is adapted
        "inputs": predict_signature,  # backward compatibility
    }

    es_config.client.update(
        index=metadata_index,
        doc_type=es_config.es_doc,
        id=es_config.es_index,
        body={
            "doc": dict(name=name, created_at=datetime.datetime.now(), **parameters),
            "doc_as_upsert": True,
        },
    )


def _interpret_dataframe(
    df: pd.DataFrame,
    pipeline: Pipeline,
    interpreter_klass: Type = DefaultInterpreterClass,
) -> pd.Series:
    """
    Apply a model interpretation to every partition dataframe

    Parameters
    ----------
    df: pd.DataFrame
        The partition DataFrame
    binary_path: str
        The model binary path
    interpreter_klass: Type
        The used interpreted class

    Returns
    -------

    A pandas Series representing the interpretations

    """

    def interpret_row(
        row: pd.Series, interpreter: SaliencyInterpreter
    ) -> Union[dict, List[dict]]:
        """Interpret a incoming dataframe row"""
        data = row.to_dict()
        interpretation = interpreter.saliency_interpret_from_json(data)
        if len(interpretation) == 0:
            return {}
        if len(interpretation) == 1:
            return interpretation[0]
        return interpretation

    interpreter = interpreter_klass(pipeline)
    return df.apply(interpret_row, interpreter=interpreter, axis=1)


def __prepare_es_index(es_config: ElasticsearchConfig, force_delete: bool):
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
        body={"mappings": {es_config.es_doc: {"dynamic_templates": dynamic_templates}}},
        ignore=400,
    )
