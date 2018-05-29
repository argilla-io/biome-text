try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
from allennlp.common import Params


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
