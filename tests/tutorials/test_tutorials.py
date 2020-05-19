from pytest_notebook.nb_regression import NBRegressionFixture
from tests import TUTORIALS_PATH
from pathlib import Path


def test_notebook():
    fixture = NBRegressionFixture(exec_timeout=50)
    fixture.diff_ignore = ("/cells/*/execution_count",)
    fixture.check(str(Path(TUTORIALS_PATH) / "test.ipynb"))
