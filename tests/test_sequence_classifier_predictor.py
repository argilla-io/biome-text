import os
import unittest

from allennlp.models import load_archive
from allennlp.service.predictors import Predictor

from recognai.service.predictors import SequenceClassifierPredictor
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

MODEL_PATH = os.path.join(TEST_RESOURCES, 'resources/models/model.tar.gz')


class SequenceClassifierPredictorTest(DaskSupportTest):

    def setUp(self):
        archive = load_archive(MODEL_PATH)
        self.predictor = Predictor.from_archive(archive, 'sequence-classifier')

    def tearDown(self):
        del self.predictor

    @unittest.skip('Update model.tar.gz configuration')
    def test_label_input(self):
        inputs = {"label": "Herbert Brandes-Siller", "Branche": "--", "category of dataset": "person",
                  "Type Info": "person"}

        result = self.predictor.predict_json(inputs)

        label = result.get("probabilities_by_class")

        assert 'person' in label
        assert 'business' in label

        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)

    @unittest.skip('Update model.tar.gz configuration')
    def test_input_that_make_me_cry(self):
        inputs = {"label": "Iem Gmbh", "Branche": "Immobilienfirmen", "category of dataset": "business",
                  "Type Info": "business record"}

        self.assertRaises(RuntimeError, self.predictor.predict_json, inputs)


