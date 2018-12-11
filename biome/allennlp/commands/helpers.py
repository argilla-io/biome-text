import yaml
from allennlp.common.checks import ConfigurationError
from allennlp.models.archival import load_archive
from typing import Optional, Dict, Any

from biome.data.utils import read_definition_from_model_spec

MODEL_FIELD = 'model'
TRAIN_DATA_FIELD = 'train_data_path'
VALIDATION_DATA_FIELD = 'validation_data_path'
TEST_DATA_FIELD = 'test_data_path'


def biome2allennlp_params(model_spec: Optional[str] = None,
                          model_binary: Optional[str] = None,
                          trainer_path: str = '',
                          train_cfg: str = '',
                          validation_cfg: str = '',
                          test_cfg: Optional[str] = None) -> Dict[str, Any]:
    if not model_binary and not model_spec:
        raise ConfigurationError('Missing parameter --spec/--binary')

    with open(trainer_path) as trainer_file:
        trainer_params = yaml.load(trainer_file)
        cfg_params = __load_from_archive(model_binary) \
            if model_binary \
            else read_definition_from_model_spec(model_spec) if model_spec else dict()

    allennlp_configuration = {
        **cfg_params,
        **trainer_params,
        TRAIN_DATA_FIELD: train_cfg,
        VALIDATION_DATA_FIELD: validation_cfg
    }

    if test_cfg and not test_cfg.isspace():
        allennlp_configuration.update({TEST_DATA_FIELD: test_cfg})

    allennlp_configuration[MODEL_FIELD] = __merge_model_params(model_binary, allennlp_configuration.get(MODEL_FIELD))

    return allennlp_configuration


def __load_from_archive(model_binary: str) -> Dict[str, Any]:
    archive = load_archive(model_binary)
    return archive.config.as_dict()


def __merge_model_params(model_location: Optional[str], model_params: Dict[str, Any]) -> Dict:
    return {**model_params, **dict(model_location=model_location)} \
        if model_location \
        else model_params
