from pathlib import Path
from typing import Any
from typing import Dict

import pytorch_lightning
import torch.nn as nn
from allennlp.common import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import Trainer
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration


def _read_configs(configurations_path: Path, section: str) -> Dict[str, Any]:
    code_blocks = {}
    with configurations_path.open() as file:
        in_section = False
        in_new_config = False

        for line in file.readlines():
            if line.startswith(f"## {section}"):
                in_section = True
            elif line.startswith("### ") and in_section:
                code_blocks[line.split(maxsplit=1)[1]] = ""
            elif line.startswith("```python") and in_section:
                in_new_config = True
            elif line.startswith("```") and in_new_config:
                in_new_config = False
            elif line.startswith("## ") and in_section:
                in_section = False

            elif in_section and in_new_config:
                key = list(code_blocks.keys())[-1]
                code_blocks[key] += line

    configurations = {}
    for name, code in code_blocks.items():
        config = {}
        exec(code, globals(), config)
        configurations[name] = config[list(config.keys())[-1]]

    return configurations


def test_pipeline_configs(configurations_path):
    configs = _read_configs(configurations_path, "Pipeline")
    for config_name, config in configs.items():
        Pipeline.from_config(config)


def test_trainer_configs(configurations_path):
    configs = _read_configs(configurations_path, "Trainer")
    pipeline = Pipeline.from_config(
        {
            "name": "test",
            "head": {"type": "TextClassification", "labels": ["pos", "neg"]},
        }
    )
    dataset = Dataset.from_dict({"text": ["test"], "label": ["pos"]})
    linear = nn.Linear(2, 2)
    for config_name, config in configs.items():
        assert isinstance(config, TrainerConfiguration)

        trainer = Trainer(
            pipeline=pipeline, train_dataset=dataset, trainer_config=config
        )
        assert isinstance(trainer.trainer, pytorch_lightning.Trainer)


def test_vocab_configs(configurations_path):
    configs = _read_configs(configurations_path, "Vocabulary")
    for config_name, config in configs.items():
        assert isinstance(config, VocabularyConfiguration)
