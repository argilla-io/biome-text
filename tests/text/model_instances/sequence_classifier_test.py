import os
import tempfile
import multiprocessing
import unittest
from time import sleep

import requests
from biome.data.utils import ENV_ES_HOSTS
from elasticsearch import Elasticsearch

from biome.text import BaseModelInstance
from biome.text.commands.predict.predict import predict
from biome.text.commands.serve.serve import serve
from biome.text.model_instances.sequence_classifier import SequenceClassifier
from tests.test_context import TEST_RESOURCES

BASE_CONFIG_PATH = os.path.join(TEST_RESOURCES, "resources/models/sequence_classifier")


class SequenceClassifierTest(unittest.TestCase):
    output_dir = tempfile.mkdtemp()
    model_archive = os.path.join(output_dir, "model.tar.gz")

    name = "sequence_pair_classifier"
    model_path = os.path.join(BASE_CONFIG_PATH, "model.yml")
    trainer_path = os.path.join(BASE_CONFIG_PATH, "trainer.yml")
    training_data = os.path.join(BASE_CONFIG_PATH, "train.data.yml")
    validation_data = os.path.join(BASE_CONFIG_PATH, "validation.data.yml")

    def test_model_workflow(self):
        self.check_train(SequenceClassifier)
        self.check_predict()
        self.check_serve()
        self.check_predictor()

    def check_train(self, cls_type):

        classifier = BaseModelInstance.from_config(self.model_path)
        self.assertIsInstance(classifier, cls_type)

        classifier.learn(
            trainer=self.trainer_path,
            train=self.training_data,
            validation=self.validation_data,
            output=self.output_dir,
        )

        self.assertTrue(classifier.architecture is not None)

        prediction = classifier.predict("mike Farrys")
        self.assertTrue("logits" in prediction, f"Not in {prediction}")

    def check_predict(self):
        index = self.name
        es_host = os.getenv(ENV_ES_HOSTS, "http://localhost:9200")
        predict(
            binary=self.model_archive,
            from_source=self.validation_data,
            to_sink=dict(
                index=index,
                index_recreate=True,
                type="doc",
                es_hosts=es_host,
                es_batch_size=100,
            ),
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
        sleep(2)

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
