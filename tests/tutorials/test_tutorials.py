from pytest_notebook.nb_regression import NBRegressionFixture
from tests import TUTORIALS_PATH
import os


def test_text_classifier_tutorial():
    fixture = NBRegressionFixture(exec_timeout=100)
    fixture.check(os.path.join(TUTORIALS_PATH, "Training_a_text_classifier.ipynb"))
