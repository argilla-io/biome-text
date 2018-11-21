import os

from allennlp.models import load_archive
from allennlp.service.predictors import Predictor

from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

MODEL_PATH = os.path.join(TEST_RESOURCES, 'resources/models/model.tar.gz')


class SequencePairClassifierPredictorTest(DaskSupportTest):

    def setUp(self):
        archive = load_archive(MODEL_PATH)
        self.predictor = Predictor.from_archive(archive, 'sequence_classifier')

    def tearDown(self):
        del self.predictor

    def test_batch_input(self):
        inputs = [{
            'record_1': "Herbert Brandes-Siller",
            'record_2': "Herbert Brandes-Siller",
            "gold_label": "duplicate"
        }]

        results = self.predictor.predict_batch_json(inputs)
        result = results[0]
        annotation = result.get('annotation')
        classes = annotation.get('classes')

        assert 'duplicate' in classes
        assert 'not_duplicate' in classes

        assert all(prob > 0 for _, prob in classes.items())

        assert len(results) == 1

    # @unittest.skip('Update model.tar.gz configuration')
    def test_label_input(self):
        inputs = {
            'record_1': "Herbert Brandes-Siller",
            'record_2': "Herbert Brandes-Siller",
            "gold_label": "duplicate"
        }

        result = self.predictor.predict_json(inputs)

        annotation = result.get('annotation')
        classes = annotation.get('classes')

        assert 'duplicate' in classes
        assert 'not_duplicate' in classes

        assert all(prob > 0 for _, prob in classes.items())

    def test_input_that_make_me_cry(self):
        input = {'gold_label': 'duplicated', 'record_1': "Herbert Brandes-Siller"}

        self.assertRaises(Exception, self.predictor.predict_json, input)
