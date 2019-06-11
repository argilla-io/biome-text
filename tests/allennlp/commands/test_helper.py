import os
import unittest

from biome.allennlp.commands.helpers import biome2allennlp_params, VALIDATION_DATA_FIELD, BiomeConfig

from tests.test_context import TEST_RESOURCES

WITHOUT_CUDA_DEVICE_TRAINER_PATH = os.path.join(TEST_RESOURCES, 'resources/trainers/no_cuda_device_trainer.yml')
SPEC_PATH = os.path.join(TEST_RESOURCES, 'resources/models/bert.classifier.yml')


class TestCommandHelper(unittest.TestCase):

    def test_BiomeConfig_cuda_env_overrides(self):
        cfg = BiomeConfig(model_path=SPEC_PATH, trainer_path=WITHOUT_CUDA_DEVICE_TRAINER_PATH).to_allennlp_params()
        trainer_cfg = cfg['trainer']

        assert trainer_cfg['cuda_device'] == -1, 'Wrong cuda device expected'

        os.environ['CUDA_DEVICE'] = str(3)
        cfg = BiomeConfig(model_path=SPEC_PATH, trainer_path=WITHOUT_CUDA_DEVICE_TRAINER_PATH).to_allennlp_params()
        trainer_cfg = cfg['trainer']

        assert trainer_cfg['cuda_device'] == int(os.getenv('CUDA_DEVICE')), 'Wrong cuda device expected'

    def test_biome2allennlp_params_optional_validation(self):
        cfg = biome2allennlp_params(model_spec=SPEC_PATH, trainer_path=WITHOUT_CUDA_DEVICE_TRAINER_PATH)
        assert cfg.get(VALIDATION_DATA_FIELD) is None, 'Non expected value for validation dataset'

        cfg = biome2allennlp_params(model_spec=SPEC_PATH, trainer_path=WITHOUT_CUDA_DEVICE_TRAINER_PATH,
                                    validation_cfg='bad/path')

        assert cfg.get(VALIDATION_DATA_FIELD) is not None, 'Expected value for validation dataset'
