import os
import unittest

from biome.allennlp.commands.helpers import BiomeConfig

from tests.test_context import TEST_RESOURCES

WITHOUT_CUDA_DEVICE_TRAINER_PATH = os.path.join(
    TEST_RESOURCES, "resources/trainers/no_cuda_device_trainer.yml"
)
SPEC_PATH = os.path.join(TEST_RESOURCES, "resources/models/bert.classifier.yml")


class TestCommandHelper(unittest.TestCase):
    def test_BiomeConfig_cuda_env_overrides(self):
        cfg = BiomeConfig(
            model_path=SPEC_PATH, trainer_path=WITHOUT_CUDA_DEVICE_TRAINER_PATH
        ).to_allennlp_params()
        trainer_cfg = cfg["trainer"]

        assert trainer_cfg["cuda_device"] == -1, "Wrong cuda device expected"

        os.environ["CUDA_DEVICE"] = str(3)
        cfg = BiomeConfig(
            model_path=SPEC_PATH, trainer_path=WITHOUT_CUDA_DEVICE_TRAINER_PATH
        ).to_allennlp_params()
        trainer_cfg = cfg["trainer"]

        assert trainer_cfg["cuda_device"] == int(
            os.getenv("CUDA_DEVICE")
        ), "Wrong cuda device expected"
