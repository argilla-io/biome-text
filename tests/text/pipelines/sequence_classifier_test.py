import multiprocessing
import os
import tempfile
import unittest
from time import sleep

import requests
from biome.data.utils import ENV_ES_HOSTS
from elasticsearch import Elasticsearch

from biome.text.commands.explore.explore import explore
from biome.text.commands.serve.serve import serve
from biome.text.pipelines.sequence_classifier import SequenceClassifier
from tests.test_context import TEST_RESOURCES

BASE_CONFIG_PATH = os.path.join(TEST_RESOURCES, "resources/models/sequence_classifier")


class SequenceClassifierTest(unittest.TestCase):
    output_dir = tempfile.mkdtemp()
    model_archive = os.path.join(output_dir, "model.tar.gz")

    name = "sequence_classifier"
    model_path = os.path.join(BASE_CONFIG_PATH, "model.yml")
    trainer_path = os.path.join(BASE_CONFIG_PATH, "trainer.yml")
    training_data = os.path.join(BASE_CONFIG_PATH, "train.data.yml")
    validation_data = os.path.join(BASE_CONFIG_PATH, "validation.data.yml")

    def test_model_workflow(self):
        self.check_train()
        self.check_predict()
        self.check_serve()
        self.check_predictor()

    def check_train(self):

        classifier = SequenceClassifier.from_config(self.model_path)
        self.assertIsInstance(classifier, SequenceClassifier)

        classifier.learn(
            trainer=self.trainer_path,
            train=self.training_data,
            validation=self.validation_data,
            output=self.output_dir,
        )

        self.assertTrue(classifier.model is not None)

        prediction = classifier.predict("mike Farrys")
        self.assertTrue("logits" in prediction, f"Not in {prediction}")

    def check_predict(self):
        index = self.name
        es_host = os.getenv(ENV_ES_HOSTS, "http://localhost:9200")
        explore(
            binary=self.model_archive,
            source_path=self.validation_data,
            es_host=es_host,
            es_index=index,
        )

        client = Elasticsearch(hosts=es_host, http_compress=True)
        data = client.search(index)
        self.assertIn("hits", data, msg=f"Must exists hits in es response {data}")
        self.assertTrue(len(data["hits"]) > 0, "No data indexed")

    def check_serve(self):
        port = 8000
        process = multiprocessing.Process(
            target=serve, daemon=True, kwargs=dict(binary=self.model_archive, port=port)
        )
        process.start()
        sleep(5)

        response = requests.post(
            f"http://localhost:{port}/predict", json={"tokens": "Mike Farrys"}
        )
        self.assertTrue(response.json() is not None)
        process.terminate()

    def check_predictor(self):
        predictor = SequenceClassifier.load(self.model_archive)

        def test_batch_input():
            inputs = [{"tokens": "Herbert Brandes-Siller", "label": "duplicate"}]

            results = predictor.predict_batch_json(inputs)
            result = results[0]
            classes = result.get("classes")

            for the_class in ["duplicate", "not_duplicate"]:
                self.assertIn(the_class, classes)

            self.assertTrue(all(prob > 0 for _, prob in classes.items()))
            self.assertEqual(1, len(results))

        def test_label_input():
            inputs = {"tokens": "Herbert Brandes-Siller", "label": "duplicate"}

            result = predictor.predict_json(inputs)
            classes = result.get("classes")

            for the_class in ["duplicate", "not_duplicate"]:
                self.assertIn(the_class, classes)

            self.assertTrue(all(prob > 0 for _, prob in classes.items()))

        def test_input_that_make_me_cry():
            self.assertRaises(
                Exception,
                predictor.predict_json,
                {"label": "duplicate", "record1": "Herbert Brandes-Siller"},
            )

        test_batch_input()
        test_input_that_make_me_cry()
        test_label_input()
