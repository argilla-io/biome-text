from typing import Dict

from biome.data.biome.transformations import biome_datasource_spec_to_dataset_config, is_biome_datasource_spec

DATASET_READER_FIELD = 'dataset_reader'
MODEL_FIELD = 'model'

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
from allennlp.common import Params
import json


def read_datasource_cfg(cfg: str) -> Dict:
    try:
        ds_cfg = json.loads(cfg)
        if is_biome_datasource_spec(ds_cfg):
            return biome_datasource_spec_to_dataset_config(ds_cfg)
        return ds_cfg
    except:
        return dict(path=cfg)


def is_biome_model_spec(spec: Dict) -> bool:
    return 'definition' in spec and 'topology' in spec['definition']


def read_definition_from_model_spec(path: str) -> Dict:
    with open(path) as model_file:
        model_spec = json.load(model_file)

        if is_biome_model_spec(model_spec):
            definition = model_spec['definition']['topology']
            return {
                DATASET_READER_FIELD: definition.pop('pipeline'),
                MODEL_FIELD: definition.pop('architecture')
            }
        return model_spec


def yaml_to_dict(filepath: str):
    with open(filepath) as yaml_content:
        config = yaml.load(yaml_content)
    return config


def read_params_from_file(filepath: str) -> Params:
    try:
        with open(filepath, 'r') as stream:
            params = Params(yaml.load(stream, Loader))
            stream.close()
            return params
    except:
        return Params.from_file(filepath)
