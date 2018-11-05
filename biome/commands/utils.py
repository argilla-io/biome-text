from typing import Dict, Any

from biome.spec import ModelDefinition, ModelRevision
from biome.spec.utils import to_biome_class

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
from allennlp.common import Params
import json


def read_datasource_cfg(cfg: Any) -> Dict:
    try:
        if isinstance(cfg, str):
            with open(cfg) as cfg_file:
                return json.loads(cfg_file.read())
        if isinstance(cfg, Params):
            return cfg.as_dict()
        if isinstance(cfg, Dict):
            return cfg
    except TypeError or FileNotFoundError:
        raise Exception('Missing configuration {}'.format(cfg))
    except Exception as e:
        return dict(path=cfg)


def is_biome_model_spec(spec: Dict) -> bool:
    return 'definition' in spec and 'topology' in spec['definition']


def read_definition_from_model_spec(path: str) -> Dict:
    with open(path) as model_file:
        model_data = json.load(model_file)
        try:
            model_revision = to_biome_class(data=model_data, klass=ModelRevision)
            topology = model_revision.definition.topology
            return dict(dataset_reader=topology.pipeline, model=topology.architecture)
        except:
            return model_data


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
