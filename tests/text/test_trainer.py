import pytest
from biome.text import Dataset, Pipeline, TrainerConfiguration, VocabularyConfiguration


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    data = {
        "text": ["Test", "this", "shaight", "!"],
        "label": ["0", "1", "0", "0"]
    }

    return Dataset.from_dict(data)


@pytest.fixture(scope="module")
def pipeline(dataset) -> Pipeline:
    config = {
        "name": "test_trainer",
        "features": {
            "word": {"embedding_dim": 2}
        },
        "head": {
            "type": "TextClassification",
            "labels": list(set(dataset["label"]))
        }
    }
    pl = Pipeline.from_config(config)
    pl.create_vocabulary(VocabularyConfiguration(sources=[dataset]))

    return pl


def test_use_amp(dataset, pipeline, tmp_path, capsys):
    trainer_config = TrainerConfiguration(
        num_epochs=1,
        batch_size=2,
        use_amp=True,
    )

    pipeline.train(
        output=str(tmp_path / "test_use_amp_output"),
        training=dataset,
        trainer=trainer_config
    )

    captured = capsys.readouterr()
    assert "use_amp = True" in captured.err
