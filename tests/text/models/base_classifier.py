import logging
import multiprocessing
import os
import tempfile
import unittest
from time import sleep

import requests

from biome.text import Pipeline
from biome.text.environment import ES_HOST
from elasticsearch import Elasticsearch

from biome.text.commands.explore.explore import explore
from biome.text.commands.serve.serve import serve
from biome.text.models import SequencePairClassifier, load_archive
from biome.text.predictors import get_predictor_from_archive
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

logging.basicConfig(level=logging.DEBUG)


BASE_CONFIG_PATH = os.path.join(
    TEST_RESOURCES, "resources/models/sequence_pair_classifier"
)


class BasePairClassifierTest(DaskSupportTest):
    base_config = None
    DEFAULT_REQUEST_TIMEOUT_IN_SECONDS = 10

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp()
        cls.model_archive = os.path.join(cls.output_dir, "model.tar.gz")

        cls.trainer_path = os.path.join(cls.base_config, "trainer.yml")
        cls.model_path = os.path.join(cls.base_config, "model.yml")
        cls.training_data = os.path.join(cls.base_config, "train.data.yml")
        cls.validation_data = os.path.join(cls.base_config, "validation.data.yml")

    def check_train(self, cls_type):
        pipeline = Pipeline.from_config(self.model_path)
        _ = pipeline.learn(
            output=self.output_dir,
            train=self.training_data,
            validation=self.validation_data,
            trainer=self.trainer_path,
        )
        archive = load_archive(self.model_archive)
        self.assertTrue(archive.model is not None)
        self.assertIsInstance(archive.model, cls_type)

    def check_explore(self):
        index = self.__class__.__name__.lower()
        es_host = os.getenv(ES_HOST, "http://localhost:9200")
        explore(
            binary=self.model_archive,
            source_path=self.validation_data,
            es_host=es_host,
            es_index=index,
            interpret=True,  # Enable interpret
        )

        client = Elasticsearch(hosts=es_host, http_compress=True)
        data = client.search(index)
        self.assertIn("hits", data, msg=f"Must exists hits in es response {data}")
        self.assertTrue(len(data["hits"]) > 0, "No data indexed")

    def check_serve(self):
        port = 18000
        process = multiprocessing.Process(
            target=serve, daemon=True, kwargs=dict(binary=self.model_archive, port=port)
        )
        process.start()
        sleep(5)
        try:
            response = requests.post(
                f"http://localhost:{port}/predict",
                json={"record1": "mike Farrys", "record2": "Mike Farris"},
                timeout=self.DEFAULT_REQUEST_TIMEOUT_IN_SECONDS,
            )
            self.assertTrue(response.json() is not None)
        finally:
            process.terminate()
        sleep(2)

    def check_predictor(self):
        predictor = get_predictor_from_archive(load_archive(self.model_archive))

        def test_batch_input(self):
            inputs = [
                {
                    "record1": "Herbert Brandes-Siller",
                    "record2": "Herbert Brandes-Siller",
                    "label": "duplicate",
                }
            ]

            results = predictor.predict_batch_json(inputs)
            annotation = results[0]
            classes = annotation.get("classes")

            for the_class in ["duplicate", "not_duplicate"]:
                self.assertIn(the_class, classes)

            self.assertTrue(all(prob > 0 for _, prob in classes.items()))
            self.assertEqual(1, len(results))

        def test_label_input(self):
            inputs = {
                "record1": "Herbert Brandes-Siller",
                "record2": "Herbert Brandes-Siller",
                "label": "duplicate",
            }

            annotation = predictor.predict_json(inputs)
            classes = annotation.get("classes")

            for the_class in ["duplicate", "not_duplicate"]:
                self.assertIn(the_class, classes)

            assert all(prob > 0 for _, prob in classes.items())

        def test_input_that_make_me_cry(self):
            self.assertRaises(
                Exception,
                predictor.predict_json,
                {"label": "duplicate", "record1": "Herbert Brandes-Siller"},
            )

        test_batch_input(self)
        test_input_that_make_me_cry(self)
        test_label_input(self)
