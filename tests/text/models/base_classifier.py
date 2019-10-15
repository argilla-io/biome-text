import logging
import multiprocessing
import os
import tempfile
from time import sleep

import requests
from biome.data.utils import ENV_ES_HOSTS
from elasticsearch import Elasticsearch

from biome.text.commands.learn.learn import learn
from biome.text.commands.predict.predict import predict
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

    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()
        self.model_archive = os.path.join(self.output_dir, "model.tar.gz")

        self.model_path = os.path.join(self.base_config, "model.yml")
        self.trainer_path = os.path.join(self.base_config, "trainer.yml")
        self.training_data = os.path.join(self.base_config, "train.data.yml")
        self.validation_data = os.path.join(self.base_config, "validation.data.yml")

    def model_workflow(self):
        self.check_train(SequencePairClassifier)
        self.check_predict()
        self.check_serve()
        self.check_predictor()

    def check_train(self, cls_type):

        _ = learn(
            model_spec=self.model_path,
            output=self.output_dir,
            train_cfg=self.training_data,
            validation_cfg=self.validation_data,
            trainer_path=self.trainer_path,
        )
        archive = load_archive(self.model_archive)
        self.assertTrue(archive.model is not None)
        self.assertIsInstance(archive.model, cls_type)

    def check_predict(self):
        index = self.__class__.__name__.lower()
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
            f"http://localhost:{port}/predict",
            json={"record1": "mike Farrys", "record2": "Mike Farris"},
        )
        self.assertTrue(response.json() is not None)
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
            result = results[0]
            annotation = result.get("annotation")
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

            result = predictor.predict_json(inputs)

            annotation = result.get("annotation")
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
