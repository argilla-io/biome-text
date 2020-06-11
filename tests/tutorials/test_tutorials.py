import re
from pathlib import Path

from pytest_notebook.nb_regression import NBRegressionFixture
from pytest_notebook.notebook import load_notebook, dump_notebook

from tests import TUTORIALS_PATH


def test_text_classifier_tutorial(tmp_path):
    notebook_path = Path(TUTORIALS_PATH) / "Training_a_text_classifier.ipynb"

    # adapt notebook to CI (make its execution quicker + comment lines)
    notebook = load_notebook(str(notebook_path))
    for cell in notebook["cells"]:
        if cell["source"].startswith("!pip install"):
            cell["source"] = re.sub(
                r"!pip install", r"#!pip install", cell["source"]
            )
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
