from typing import Type

from biome.data.sources import DataSource
from biome.data.utils import yaml_to_dict


def get_reader_configuration(file_path: str, forward_config_class: Type):
    cfg = yaml_to_dict(file_path)
    forward = forward_config_class(**cfg.pop("forward", dict()))
    data_source = DataSource.from_cfg(cfg, cfg_file=file_path)
    return data_source, forward
