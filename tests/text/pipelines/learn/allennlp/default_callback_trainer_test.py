import unittest

from allennlp.common import Params

from biome.text.pipelines._impl.allennlp.learn.default_callback_trainer import (
    DefaultCallbackTrainer,
)


class DefaultCallbackTrainerTest(unittest.TestCase):
    def test_disable_validate(self):
        callback_config = DefaultCallbackTrainer._callbacks_configuration(
            trainer_params=Params({}), validate=False
        )
        self.assertNotIn(None, callback_config)
        self.assertNotIn("validate", callback_config)

    def test_enable_validate(self):
        callback_config = DefaultCallbackTrainer._callbacks_configuration(
            trainer_params=Params({}), validate=True
        )
        self.assertNotIn(None, callback_config)
        self.assertIn("validate", callback_config)

    def test_disable_evaluate(self):
        callback_config = DefaultCallbackTrainer._callbacks_configuration(
            trainer_params=Params({}), evaluate=False
        )
        self.assertNotIn(None, callback_config)
        self.assertNotIn("evaluate", callback_config)

    def test_enable_evaluate(self):
        callback_config = DefaultCallbackTrainer._callbacks_configuration(
            trainer_params=Params({}), evaluate=True
        )
        self.assertIn("evaluate", callback_config)


if __name__ == "__main__":
    unittest.main()
