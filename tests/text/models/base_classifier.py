import logging
import multiprocessing
import os
import tempfile
from time import sleep

import pytest
import requests
import torch
from allennlp.modules import Seq2VecEncoder
from allennlp.modules import TextFieldEmbedder
from biome.text import Pipeline
from biome.text.commands.explore.explore import explore
from biome.text.commands.serve.serve import serve
from biome.text.environment import ES_HOST
from biome.text.pipelines._impl.allennlp.models import SequenceClassifierBase
from biome.text.pipelines._impl.allennlp.models import load_archive
from biome.text.pipelines._impl.allennlp.predictors.utils import (
    get_predictor_from_archive,
)
from elasticsearch import Elasticsearch

from tests.test_context import TEST_RESOURCES
from tests.test_support import DaskSupportTest

logging.basicConfig(level=logging.DEBUG)


BASE_CONFIG_PATH = os.path.join(
    TEST_RESOURCES, "resources/models/sequence_pair_classifier"
)


def test_loss_weights(tokens_labels_vocab):
    encoder = Seq2VecEncoder()
    encoder.get_output_dim = lambda: 1
    model = SequenceClassifierBase(
        vocab=tokens_labels_vocab,
        text_field_embedder=TextFieldEmbedder(),
        seq2vec_encoder=encoder,
        loss_weights={"label0": 0.0, "label1": 1.0},
    )

    input_tensor = torch.tensor([[1.0, 1.0]])
    class_tensor0 = torch.tensor([0], dtype=torch.long)
    class_tensor1 = torch.tensor([1], dtype=torch.long)

    assert model._loss(input=input_tensor, target=class_tensor0) == torch.tensor(0)
    assert model._loss(input=input_tensor, target=class_tensor1) == -torch.log(
        torch.tensor(0.5)
    )

    with pytest.raises(KeyError):
        SequenceClassifierBase(
            vocab=tokens_labels_vocab,
            text_field_embedder=TextFieldEmbedder(),
            seq2vec_encoder=encoder,
            loss_weights={"label0": 0.0, "missing_label": 1.0},
        )
    with pytest.raises(RuntimeError):
        SequenceClassifierBase(
            vocab=tokens_labels_vocab,
            text_field_embedder=TextFieldEmbedder(),
            seq2vec_encoder=encoder,
            loss_weights={"label0": 0.0, "label1": 1.0, "extra_label": 0.0},
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
            target=serve,
            daemon=True,
            kwargs=dict(binary=self.model_archive, port=port),
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
