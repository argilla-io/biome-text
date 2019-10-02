import logging
import os
import tempfile
import threading
from time import sleep

import requests
from biome.data.utils import ENV_ES_HOSTS
from elasticsearch import Elasticsearch

from biome.text.commands.learn.learn import learn
from biome.text.commands.predict.predict import predict
from biome.text.commands.serve.serve import serve
from biome.text.models import SequencePairClassifier, load_archive
from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

logging.basicConfig(level=logging.DEBUG)


BASE_CONFIG_PATH = os.path.join(
    TEST_RESOURCES, "resources/definitions/sequence_pair_classifier"
)


class SequencePairClassifierTest(DaskSupportTest):
    output_dir = tempfile.mkdtemp()
    model_archive = os.path.join(output_dir, "model.tar.gz")

    name = "sequence_pair_classifier"
    model_path = os.path.join(BASE_CONFIG_PATH, "model.yml")
    trainer_path = os.path.join(BASE_CONFIG_PATH, "trainer.yml")
    training_data = os.path.join(BASE_CONFIG_PATH, "train.data.yml")
    validation_data = os.path.join(BASE_CONFIG_PATH, "validation.data.yml")

    def test_model_workflow(self):
        self.check_train(SequencePairClassifier)
        self.check_predict()
        self.check_serve()

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
        thread = threading.Thread(
            target=serve, daemon=True, kwargs=dict(binary=self.model_archive, port=port)
        )
        thread.start()
        sleep(5)

        response = requests.post(
            f"http://localhost:{port}/predict",
            json={"record1": "mike Farrys", "record2": "Mike Farris"},
        )
        self.assertTrue(response.json() is not None)
