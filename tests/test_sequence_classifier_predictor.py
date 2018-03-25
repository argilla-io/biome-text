import os
import unittest

from allennlp.models import load_archive
from allennlp.service.predictors import Predictor

from recognai.service.predictors import SequenceClassifierPredictor
from tests.test_context import TEST_RESOURCES

MODEL_PATH = os.path.join(TEST_RESOURCES, 'resources/models/model.tar.gz')


class SequenceClassifierPredictorTest(unittest.TestCase):
    _ = SequenceClassifierPredictor  # Avoid remove import when auto save

    def test_label_input(self):
        inputs = {"label": "Herbert Brandes-Siller", "Branche": "--", "category of dataset": "person",
                  "Type Info": "person"}

        archive = load_archive(MODEL_PATH)
        predictor = Predictor.from_archive(archive, 'sequence-classifier')

        result = predictor.predict_json(inputs)

        label = result.get("probabilities_by_class")
        
        assert 'person' in label
        assert 'business' in label

        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)

    def test_input_that_make_me_cry(self):
        inputs = {"label": "Iem Gmbh", "Branche": "Immobilienfirmen", "category of dataset": "business",
                  "Type Info": "business record"}

        archive = load_archive(MODEL_PATH)
        predictor = Predictor.from_archive(archive, 'sequence-classifier')

        self.assertRaises(RuntimeError, predictor.predict_json, inputs)


if __name__ == '__main__':
    unittest.main()
