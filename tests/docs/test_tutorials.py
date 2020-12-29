import re

import pytest
from pytest_notebook.nb_regression import NBRegressionFixture
from pytest_notebook.notebook import dump_notebook
from pytest_notebook.notebook import load_notebook

pytestmark = pytest.mark.skip(
    reason="The pytest-notebook package is not actively maintained and "
    "the tutorial tests are quite heavy on resources. "
    "The idea is to run those tests locally and manually from time to time."
    "THESE TESTS ARE ALSO OUT OF DATE ... :/"
)


def test_text_classifier_tutorial(tmp_path, tutorials_path):
    notebook_path = tutorials_path / "Training_a_text_classifier.ipynb"

    # adapt notebook to CI (make its execution quicker + comment lines)
    notebook = load_notebook(str(notebook_path))
    for cell in notebook["cells"]:
        if cell["source"].startswith("!pip install"):
            cell["source"] = re.sub(r"!pip install", r"#!pip install", cell["source"])
        if cell["source"].startswith("trainer_config ="):
            cell["source"] = re.sub(
                r"num_epochs=[0-9][0-9]?", r"num_epochs=1", cell["source"]
            )
        if cell["source"].startswith("pl.train("):
            cell["source"] = re.sub(
                r"training=train_ds", r"training=valid_ds", cell["source"]
            )
        if cell["source"].startswith("pl_trained.explore"):
            cell["source"] = re.sub(
                r"pl_trained.explore", r"#pl_trained.explore", cell["source"]
            )

    # dump adapted notebook
    mod_notebook_path = tmp_path / notebook_path.name
    with mod_notebook_path.open("w") as file:
        file.write(str(dump_notebook(notebook)))

    # test adapted notebook
    fixture = NBRegressionFixture(exec_timeout=100)
    fixture.check(str(mod_notebook_path))


def test_slot_filling_tutorial(tmp_path, tutorials_path):
    notebook_path = tutorials_path / "Training_a_sequence_tagger_for_Slot_Filling.ipynb"

    # adapt notebook to CI (make its execution quicker + comment lines)
    notebook = load_notebook(str(notebook_path))
    for cell in notebook["cells"]:
        if cell["source"].startswith("!pip install"):
            cell["source"] = re.sub(r"!pip install", r"#!pip install", cell["source"])
        if cell["source"].startswith(
            "from biome.text.configuration import FeaturesConfiguration"
        ):
            cell["source"] = re.sub(
                r"https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
                r"https://biome-tutorials-data.s3-eu-west-1.amazonaws.com/token_classifier/wiki-news-300d-1M.head.vec",
                cell["source"],
            )
        if cell["source"].startswith("trainer_config ="):
            cell["source"] = re.sub(
                r"TrainerConfiguration\(\)",
                r"TrainerConfiguration(num_epochs=1)",
                cell["source"],
            )
        if cell["source"].startswith("pl.train("):
            cell["source"] = re.sub(
                r"pl.train",
                r"from biome.text.configuration import TrainerConfiguration\npl.train",
                cell["source"],
            )
            cell["source"] = re.sub(
                r"training=train_ds",
                r"training=valid_ds",
                cell["source"],
            )
            cell["source"] = re.sub(
                r"test=test_ds,",
                r"test=test_ds, trainer=TrainerConfiguration(num_epochs=1)",
                cell["source"],
            )

    # dump adapted notebook
    mod_notebook_path = tmp_path / notebook_path.name
    with mod_notebook_path.open("w") as file:
        file.write(str(dump_notebook(notebook)))

    # test adapted notebook
    fixture = NBRegressionFixture(exec_timeout=200)
    fixture.check(str(mod_notebook_path))
