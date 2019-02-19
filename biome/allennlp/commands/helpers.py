import os
from copy import deepcopy
from typing import Optional, Dict, Any

import yaml

from biome.allennlp.data.transformations import biome_datasource_spec_to_dataset_config
from biome.spec import ModelDefinition
from biome.spec.utils import to_biome_class

CUDA_DEVICE_FIELD = 'cuda_device'
MODEL_FIELD = 'model'
TRAINER_FIELD = 'trainer'
TRAIN_DATA_FIELD = 'train_data_path'
VALIDATION_DATA_FIELD = 'validation_data_path'
TEST_DATA_FIELD = 'test_data_path'
EVALUATE_ON_TEST_FIELD = 'evaluate_on_test'

from biome.data.utils import read_datasource_cfg


def read_datasource_configuration(config: Any) -> Dict:
    cfg = read_datasource_cfg(config)
    try:
        return biome_datasource_spec_to_dataset_config(deepcopy(cfg))
    except Exception as e:
        return cfg


def biome2allennlp_params(model_spec: Optional[str] = None,
                          trainer_path: Optional[str] = None,
                          vocab_path: Optional[str] = None,
                          train_cfg: str = '',
                          validation_cfg: Optional[str] = None,
                          test_cfg: Optional[str] = None) -> Dict[str, Any]:
    def load_yaml_config(from_path: Optional[str]) -> Dict[str, Any]:
        if not from_path:
            return dict()
        with open(from_path) as trainer_file:
            return yaml.load(trainer_file)

    def read_definition_from_model_spec(path: str) -> Dict:
        with open(path) as model_file:
            model_data = yaml.load(model_file)
            try:
                model_definition = to_biome_class(data=model_data, klass=ModelDefinition)
                topology = model_definition.topology
                return dict(dataset_reader=topology.pipeline, model=topology.architecture)
            except Exception as e:
                print(e)
                return model_data

    cfg_params = read_definition_from_model_spec(model_spec) \
        if model_spec \
        else dict()

    trainer_cfg = load_yaml_config(trainer_path)
    cuda_dive = trainer_cfg[TRAINER_FIELD].get(CUDA_DEVICE_FIELD, int(os.getenv(CUDA_DEVICE_FIELD.upper(), -1)))
    trainer_cfg[TRAINER_FIELD][CUDA_DEVICE_FIELD] = cuda_dive

    vocab_cfg = load_yaml_config(vocab_path)
    allennlp_configuration = {
        **cfg_params,
        **trainer_cfg,
        **vocab_cfg,
        TRAIN_DATA_FIELD: train_cfg,
        EVALUATE_ON_TEST_FIELD: True,  # When no test data is provided, this param is ignored
    }

    for cfg, field in [(validation_cfg, VALIDATION_DATA_FIELD), (test_cfg, TEST_DATA_FIELD)]:
        if cfg and not cfg.isspace():
            allennlp_configuration.update({field: cfg})

    return allennlp_configuration
