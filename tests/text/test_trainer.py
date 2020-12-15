import pytest
import torch

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    data = {"text": ["Test", "this", "shaight", "!"], "label": ["0", "1", "0", "0"]}

    return Dataset.from_dict(data)


@pytest.fixture(scope="module")
def pipeline(dataset) -> Pipeline:
    config = {
        "name": "test_trainer",
        "features": {"word": {"embedding_dim": 2}},
        "head": {"type": "TextClassification", "labels": list(set(dataset["label"]))},
    }
    pl = Pipeline.from_config(config)

    return pl


@pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="Using AMP requires a cuda device"
)
def test_use_amp(dataset, pipeline, tmp_path, capsys):
    trainer_config = TrainerConfiguration(
        num_epochs=1,
        batch_size=2,
        use_amp=True,
    )

    pipeline.train(
        output=str(tmp_path / "test_use_amp_output"),
        training=dataset,
        trainer=trainer_config,
    )

    captured = capsys.readouterr()
    assert "use_amp = True" in captured.err
